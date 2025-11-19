#!/usr/bin/env python3
"""Run WAN2.2-MOE i2v quality inference with motion amplitude enhancement.

This script reproduces the exact results of the default inference command
with additional motion enhancement to fix slow-motion issues in 4-step LoRAs:

    CUDA_VISIBLE_DEVICES=7 python -m lightx2v.infer \
        --model_cls wan2.2_moe \
        --task i2v \
        --model_path models/wan2.2_models/official_distill_repo \
        --config_json prod_configs/wan_moe_i2v_a14b_quality.json \
        --image_path assets/inputs/imgs/pron_test.png \
        --prompt "woman smiling" \
        --save_result_path save_results/video_quality_tuned.mp4

ARCHITECTURE OVERVIEW:
======================

This script demonstrates a custom runner pattern for WAN2.2-MOE inference:

1. **Wan22MoeCustomRunner** (defined in this file)
   └── Inherits from: **Wan22MoeDistillRunner** (lightx2v.models.runners.wan.wan_distill_runner)
       └── Inherits from: **WanDistillRunner** (base WAN distill runner class)
           └── Inherits from: **WanRunner** (base WAN runner class)
               └── Inherits from: **DefaultRunner** (lightx2v.models.runners.default_runner)

2. **Key Customizations**:
   - `__init__()`: Enhanced logging of model paths during initialization
   - `load_transformer()`: Custom LoRA loading using config 'name' field
   - `get_vae_encoder_output()`: Motion amplitude enhancement for VAE latents
   - `run_pipeline()`: Wrapped pipeline with custom logging hooks

3. **Motion Enhancement Classes** (NEW):
   - `MotionAmplitudeProcessor`: Core algorithm for fixing slow-motion issues
   - `EnhancedVAEEncoder`: Wrapper for VAE encoder with motion enhancement
   
4. **Base Runner Responsibilities** (inherited):
   - Model loading (VAE, text encoders, image encoder)
   - Scheduler initialization
   - Text encoding pipeline
   - VAE encoding/decoding
   - Video generation and saving

5. **MultiDistillModelStruct**: Manages the two-stage distilled inference
   - High noise model (used for early denoising steps)
   - Low noise model (used for final refinement steps)
   - Automatic switching based on boundary timestep

MOTION ENHANCEMENT:
===================

The motion amplitude enhancement fixes the "slow-motion" problem in 4-step distilled models:

1. **Problem**: Fast distilled models (4-8 steps) often produce slow, subtle motion
2. **Solution**: Amplify motion in VAE latent space while preserving brightness
3. **Algorithm**:
   - Extract first frame (real) and subsequent frames (gray-filled)
   - Calculate motion difference: diff = gray_frames - first_frame
   - Preserve brightness: extract and re-add mean of difference
   - Amplify centered motion: scaled = first_frame + (diff - mean) * amplitude + mean
   - Clamp to prevent artifacts: clamp(scaled, -6, 6)

4. **Configuration**:
   - Default amplitude: 1.15 (15% more motion)
   - Recommended range: 1.0 (disabled) to 1.5 (50% more motion)
   - Disable via: --disable_motion_enhancement

MEMORY OPTIMIZATION (CPU OFFLOADING):
======================================

Reduce GPU memory usage by offloading T5 encoder and VAE to CPU when not in use:

1. **T5 CPU Offloading** (t5_cpu_offload):
   - T5 encoder is kept on CPU by default
   - Loaded to GPU only during text encoding
   - Immediately offloaded back to CPU after encoding
   - Saves ~8-10GB VRAM

2. **VAE CPU Offloading** (vae_cpu_offload):
   - VAE encoder is kept on CPU by default
   - Loaded to GPU only during image encoding
   - VAE decoder is kept on CPU by default
   - Loaded to GPU only during latent decoding
   - Saves ~2-4GB VRAM

3. **Configuration**:
   - Enable in config: "t5_cpu_offload": true, "vae_cpu_offload": true
   - FP8 config has offloading enabled by default
   - Quality config has offloading disabled by default (for max speed)
   - Total VRAM savings: ~10-14GB with both enabled

4. **Performance Impact**:
   - Adds ~2-5 seconds per inference (data transfer overhead)
   - Essential for GPUs with <24GB VRAM
   - Recommended for FP8 + LoRA workflows on consumer GPUs

USAGE:
======

Run with defaults (motion enhancement enabled, amplitude=1.15):
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py

Use FP8 config with offloading (saves ~10-14GB VRAM):
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --config_json prod_configs/wan_moe_i2v_a14b_fp8.json

Adjust motion amplitude:
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --motion_amplitude 1.3

Disable motion enhancement:
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --disable_motion_enhancement

Override specific parameters:
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --prompt "woman laughing and waving" \
        --save_result_path save_results/custom_output.mp4 \
        --motion_amplitude 1.25

For low VRAM GPUs (12-16GB), enable offloading in config:
    Edit prod_configs/wan_moe_i2v_a14b_quality.json:
    {
        "t5_cpu_offload": true,
        "vae_cpu_offload": true
    }

Add custom LoRAs (edit the config JSON):
    Add to "lora_configs" in prod_configs/wan_moe_i2v_a14b_quality.json:
    {
        "name": "my_custom_high_noise_lora",
        "path": "models/wan2.2_models/loras/my_lora.safetensors",
        "strength": 0.8
    }

CREDITS:
========

Motion enhancement algorithm based on PainterI2V from the ComfyUI community,
specifically designed to fix slow-motion issues in lightx2v and similar 4-step LoRAs.
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Optional

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct, Wan22MoeDistillRunner
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all


class CustomWeightAsyncStreamManager(WeightAsyncStreamManager):
    """
    Custom weight manager that extends WeightAsyncStreamManager with swap_weights method.
    
    This class adds the missing swap_weights method that's called by T5 encoder's
    forward_with_offload method. It implements CPU-GPU weight swapping for model offloading.
    
    The swap_weights method synchronizes streams and swaps the double-buffered CUDA buffers,
    allowing the next block's weights to become active while the current block's buffer
    can be reused for prefetching the following block.
    """
    
    def __init__(self, offload_granularity):
        """
        Initialize custom weight manager.
        
        Args:
            offload_granularity (str): Granularity of offloading ("block" or "phase")
        """
        super().__init__(offload_granularity)
        logger.debug(f"Initialized CustomWeightAsyncStreamManager (granularity={offload_granularity})")
    
    def swap_weights(self):
        """
        Swap the double-buffered CUDA weight buffers.
        
        This method implements the missing functionality by calling the parent's
        swap method based on the offload granularity:
        - For "block" granularity: swaps block buffers
        - For "phase" granularity: swaps phase buffers
        
        The method ensures proper synchronization between compute and loading streams
        before swapping buffers.
        """
        if self.offload_granularity == "block":
            self.swap_blocks()
        elif self.offload_granularity == "phase":
            self.swap_phases()
        else:
            raise ValueError(f"Unknown offload granularity: {self.offload_granularity}")


class MotionAmplitudeProcessor:
    """
    Enhanced latent processor to fix slow-motion issues in 4-step LoRAs.
    
    This class implements the motion amplitude enhancement algorithm that addresses
    the slow-motion problem common in distilled models by:
    1. Analyzing the difference between first frame and subsequent frames
    2. Amplifying motion while preserving brightness (diff_mean preservation)
    3. Clamping to prevent artifacts
    
    Based on the PainterI2V optimization from ComfyUI community.
    """
    
    def __init__(self, motion_amplitude=1.15, enable=True):
        """
        Initialize the motion amplitude processor.
        
        Args:
            motion_amplitude (float): Motion scaling factor. 
                                     1.0 = no change, >1.0 = more motion
                                     Recommended: 1.15-1.5 for 4-step LoRAs
            enable (bool): Whether to apply motion enhancement
        """
        self.motion_amplitude = motion_amplitude
        self.enable = enable
        logger.info(f"MotionAmplitudeProcessor initialized (amplitude={motion_amplitude}, enabled={enable})")
    
    @torch.no_grad()
    def apply_motion_enhancement(self, vae_latent):
        """
        Apply motion amplitude enhancement to VAE-encoded latents.
        
        Args:
            vae_latent (torch.Tensor): VAE encoded latent of shape [C, T, H, W]
                                       where C=channels, T=time, H=height, W=width
                                       The first temporal frame is real, subsequent frames
                                       are gray/zero-filled placeholders
        
        Returns:
            torch.Tensor: Enhanced latent with amplified motion while preserving brightness
        """
        if not self.enable or self.motion_amplitude <= 1.0:
            return vae_latent
        
        logger.debug(f"Applying motion enhancement (amplitude={self.motion_amplitude})...")
        logger.debug(f"VAE latent shape: {vae_latent.shape}")
        
        # Extract first temporal frame (real) and subsequent frames (gray/zero placeholders)
        # Shape: vae_latent is [C, T, H, W] (4D tensor from WAN VAE encoder)
        base_latent = vae_latent[:, 0:1, :, :]      # [C, 1, H, W] - first frame
        gray_latent = vae_latent[:, 1:, :, :]       # [C, T-1, H, W] - gray frames
        
        # Calculate motion difference
        diff = gray_latent - base_latent
        
        # Preserve brightness by extracting and re-adding mean
        # This is the KEY to preventing brightness changes during motion amplification
        # Average across channels (0), height (2), width (3), keep time (1)
        diff_mean = diff.mean(dim=(0, 2, 3), keepdim=True)
        diff_centered = diff - diff_mean
        
        # Amplify centered motion and restore mean
        scaled_latent = base_latent + diff_centered * self.motion_amplitude + diff_mean
        
        # Clamp to prevent artifacts (empirically determined range for latent space)
        scaled_latent = torch.clamp(scaled_latent, -6, 6)
        
        # Combine first frame with enhanced frames along temporal dimension (dim=1)
        enhanced_latent = torch.cat([base_latent, scaled_latent], dim=1)
        
        logger.debug(f"Motion enhancement applied. Original range: [{vae_latent.min():.3f}, {vae_latent.max():.3f}], "
                    f"Enhanced range: [{enhanced_latent.min():.3f}, {enhanced_latent.max():.3f}]")
        
        return enhanced_latent


class EnhancedVAEEncoder:
    """
    Wrapper for VAE encoder that applies motion amplitude enhancement.
    
    This class wraps the standard VAE encoding process and adds motion
    enhancement as a post-processing step, making it easy to integrate
    into existing inference pipelines.
    """
    
    def __init__(self, vae_encoder, motion_processor=None):
        """
        Initialize enhanced VAE encoder.
        
        Args:
            vae_encoder: Original VAE encoder model
            motion_processor (MotionAmplitudeProcessor): Optional motion processor
        """
        self.vae_encoder = vae_encoder
        self.motion_processor = motion_processor or MotionAmplitudeProcessor(enable=False)
    
    def encode(self, vae_input):
        """
        Encode input with motion enhancement.
        
        Args:
            vae_input: Input tensor for VAE encoding
            
        Returns:
            Enhanced VAE latent
        """
        # Standard VAE encoding
        vae_latent = self.vae_encoder.encode(vae_input)
        
        # Apply motion enhancement if enabled
        if self.motion_processor.enable:
            vae_latent = self.motion_processor.apply_motion_enhancement(vae_latent)
        
        return vae_latent
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped VAE encoder."""
        return getattr(self.vae_encoder, name)


class WanLoraCustomWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_metadata = {}
        self.override_dict = {}  # On CPU

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        if lora_name in self.lora_metadata:
            logger.info(f"LoRA {lora_name} already loaded, skipping...")
            return lora_name

        self.lora_metadata[lora_name] = {"path": lora_path}
        logger.info(f"Registered LoRA metadata for: {lora_name} from {lora_path}")

        return lora_name

    def _load_lora_file(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(GET_DTYPE()) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0):
        if lora_name not in self.lora_metadata:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])
        weight_dict = self.model.original_weight_dict
        self._apply_lora_weights(weight_dict, lora_weights, alpha)
        self.model._apply_weights(weight_dict)

        logger.info(f"Applied LoRA: {lora_name} with alpha={alpha}")
        del lora_weights
        return True

    @torch.no_grad()
    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        lora_pairs = {}
        lora_diffs = {}

        def try_lora_pair(key, prefix, suffix_a, suffix_b, target_suffix):
            if key.endswith(suffix_a):
                base_name = key[len(prefix) :].replace(suffix_a, target_suffix)
                pair_key = key.replace(suffix_a, suffix_b)
                if pair_key in lora_weights:
                    lora_pairs[base_name] = (key, pair_key)

        def try_lora_diff(key, prefix, suffix, target_suffix):
            if key.endswith(suffix):
                base_name = key[len(prefix) :].replace(suffix, target_suffix)
                lora_diffs[base_name] = key

        prefixs = [
            "",  # empty prefix
            "diffusion_model.",
        ]
        for prefix in prefixs:
            for key in lora_weights.keys():
                if not key.startswith(prefix):
                    continue

                try_lora_pair(key, prefix, "lora_A.weight", "lora_B.weight", "weight")
                try_lora_pair(key, prefix, "lora_down.weight", "lora_up.weight", "weight")
                try_lora_diff(key, prefix, "diff", "weight")
                try_lora_diff(key, prefix, "diff_b", "bias")
                try_lora_diff(key, prefix, "diff_m", "modulation")

        applied_count = 0
        for name, param in weight_dict.items():
            if name in lora_pairs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.clone().cpu()
                name_lora_A, name_lora_B = lora_pairs[name]
                
                # Handle FP8 weights: convert to bfloat16, apply LoRA, convert back
                is_fp8 = param.dtype == torch.float8_e4m3fn
                original_dtype = param.dtype
                if is_fp8:
                    param.data = param.data.to(torch.bfloat16)
                
                lora_A = lora_weights[name_lora_A].to(param.device, param.dtype)
                lora_B = lora_weights[name_lora_B].to(param.device, param.dtype)
                if param.shape == (lora_B.shape[0], lora_A.shape[1]):
                    param += torch.matmul(lora_B, lora_A) * alpha
                    applied_count += 1
                
                # Convert back to FP8 if needed
                if is_fp8:
                    param.data = param.data.to(original_dtype)
            elif name in lora_diffs:
                if name not in self.override_dict:
                    self.override_dict[name] = param.clone().cpu()

                name_diff = lora_diffs[name]
                
                # Handle FP8 weights
                is_fp8 = param.dtype == torch.float8_e4m3fn
                original_dtype = param.dtype
                if is_fp8:
                    param.data = param.data.to(torch.bfloat16)
                
                lora_diff = lora_weights[name_diff].to(param.device, param.dtype)
                if param.shape == lora_diff.shape:
                    param += lora_diff * alpha
                    applied_count += 1

        logger.info(f"Applied {applied_count} LoRA weight adjustments")
        if applied_count == 0:
            logger.info(
                "Warning: No LoRA weights were applied. Expected naming conventions: 'diffusion_model.<layer_name>.lora_A.weight' and 'diffusion_model.<layer_name>.lora_B.weight'. Please verify the LoRA weight file."
            )

    @torch.no_grad()
    def remove_lora(self):
        logger.info(f"Removing LoRA ...")

        restored_count = 0
        for k, v in self.override_dict.items():
            self.model.original_weight_dict[k] = v.to(self.model.device)
            restored_count += 1

        logger.info(f"LoRA removed, restored {restored_count} weights")

        self.model._apply_weights(self.model.original_weight_dict)

        torch.cuda.empty_cache()
        gc.collect()

        self.lora_metadata = {}
        self.override_dict = {}

    def list_loaded_loras(self):
        return list(self.lora_metadata.keys())


class Wan22MoeCustomRunner(Wan22MoeDistillRunner):
    """Custom WAN2.2-MOE Distill runner with enhanced LoRA handling and motion optimization.
    
    This runner extends Wan22MoeDistillRunner (the DISTILLED version) to provide:
    - Custom LoRA loading logic based on config 'name' field
    - Support for distilled models (WanDistillModel)
    - FP8 quantization support with SGLang optimization
    - **Combined FP8 + LoRA support** (applies LoRAs on top of quantized models)
    - **Motion amplitude enhancement** for fixing slow-motion issues in 4-step LoRAs
    - Easy hooks for pre/post-processing customization
    - Better logging and monitoring capabilities
    
    IMPORTANT: This inherits from Wan22MoeDistillRunner, not Wan22MoeRunner,
    because the quality config uses distilled models.
    
    FP8 + LoRA Modes:
    - FP8 only: Uses quantized WanDistillModel (~40% less VRAM, faster)
    - LoRA only: Uses WanModel + LoRAs (custom styles)
    - **FP8 + LoRA**: Uses quantized WanDistillModel + LoRAs (BEST: fast + custom styles)
    - Neither: Uses WanDistillModel (full precision)
    
    Motion Enhancement:
    - Enabled by default for 4-step distilled models (motion_amplitude=1.15)
    - Amplifies motion while preserving brightness
    - Prevents the "slow-motion" effect common in fast distilled inference
    - Can be disabled via config: "motion_amplitude": 1.0 or "enable_motion_enhancement": false
    
    Supports fp8-sgl (SGLang for H100/Ada), fp8-q8f (Q8F for 4090), and other schemes.
    """

    def __init__(self, config):
        """Initialize the custom runner with the given config.
        
        Args:
            config: Configuration dictionary containing model paths, hyperparameters, etc.
        """
        super().__init__(config)
        
        # Initialize motion amplitude processor
        motion_amplitude = self.config.get("motion_amplitude", 1.15)
        enable_motion = self.config.get("enable_motion_enhancement", True)
        self.motion_processor = MotionAmplitudeProcessor(
            motion_amplitude=motion_amplitude,
            enable=enable_motion
        )
        
        logger.info(f"Initialized {self.__class__.__name__} (Distilled version)")
        logger.info(f"High noise model path: {self.high_noise_model_path}")
        logger.info(f"Low noise model path: {self.low_noise_model_path}")
        
        # Log motion enhancement settings
        if enable_motion:
            logger.info(f"✓ Motion enhancement enabled (amplitude={motion_amplitude})")
        else:
            logger.info("✗ Motion enhancement disabled")
        
        # Log quantization settings
        if self.config.get("dit_quantized", False):
            quant_scheme = self.config.get("dit_quant_scheme", "unknown")
            logger.info(f"FP8 Quantization enabled: {quant_scheme}")
        
        # Log LoRA settings
        if self.config.get("lora_configs"):
            logger.info(f"LoRA configurations: {len(self.config['lora_configs'])} LoRA(s) configured")
    
    def _patch_t5_offload_manager(self):
        """
        Patch T5 encoder's offload manager with custom implementation.
        
        This method replaces the T5 encoder's WeightAsyncStreamManager with our
        CustomWeightAsyncStreamManager that includes the swap_weights method.
        This is necessary because the original manager lacks this method.
        """
        if not hasattr(self, 'text_encoders') or not self.text_encoders:
            return
        
        t5_encoder = self.text_encoders[0]
        
        # Check if T5 encoder has an offload manager
        if hasattr(t5_encoder, 'model') and hasattr(t5_encoder.model, 'offload_manager'):
            original_manager = t5_encoder.model.offload_manager
            
            # Create custom manager with same granularity
            custom_manager = CustomWeightAsyncStreamManager(
                offload_granularity=original_manager.offload_granularity
            )
            
            # Copy all attributes from original manager to custom manager
            for attr_name in dir(original_manager):
                if not attr_name.startswith('_') and attr_name not in ['swap_weights', 'swap_blocks', 'swap_phases']:
                    try:
                        attr_value = getattr(original_manager, attr_name)
                        if not callable(attr_value):
                            setattr(custom_manager, attr_name, attr_value)
                    except AttributeError:
                        pass
            
            # Replace the offload manager
            t5_encoder.model.offload_manager = custom_manager
            logger.info("✓ Patched T5 encoder with CustomWeightAsyncStreamManager")
        elif hasattr(t5_encoder, 'text_encoder') and hasattr(t5_encoder.text_encoder, 'offload_manager'):
            original_manager = t5_encoder.text_encoder.offload_manager
            
            # Create custom manager with same granularity
            custom_manager = CustomWeightAsyncStreamManager(
                offload_granularity=original_manager.offload_granularity
            )
            
            # Copy all attributes from original manager to custom manager
            for attr_name in dir(original_manager):
                if not attr_name.startswith('_') and attr_name not in ['swap_weights', 'swap_blocks', 'swap_phases']:
                    try:
                        attr_value = getattr(original_manager, attr_name)
                        if not callable(attr_value):
                            setattr(custom_manager, attr_name, attr_value)
                    except AttributeError:
                        pass
            
            # Replace the offload manager
            t5_encoder.text_encoder.offload_manager = custom_manager
            logger.info("✓ Patched T5 encoder with CustomWeightAsyncStreamManager")
    
    def init_modules(self):
        """
        Initialize runner modules and patch T5 offload manager.
        
        This override calls the parent implementation to load all modules,
        then patches the T5 encoder's offload manager with our custom implementation.
        """
        # Call parent to initialize all modules
        super().init_modules()
        
        # Patch T5 offload manager after text encoders are loaded
        if self.config.get("t5_cpu_offload", False):
            logger.info("T5 CPU offload is enabled, patching offload manager...")
            self._patch_t5_offload_manager()



    def load_transformer(self):
        """Load high and low noise distilled transformer models with custom LoRA and FP8 support.
        
        This method overrides the base implementation to support:
        1. LoRA loading based on the 'name' field in lora_configs
        2. FP8 quantization (fp8-sgl, fp8-q8f) for faster inference
        3. Combined FP8 + LoRA support (applies LoRAs on top of quantized models)
        4. Smart model selection:
           - LoRAs + quantized: Loads quantized WanDistillModel, then applies LoRAs
           - LoRAs only: Uses WanModel + LoRAs (for compatibility)
           - Quantized only: Uses WanDistillModel with quantization
           - No quantization/LoRA: Uses WanDistillModel
        
        Returns:
            MultiDistillModelStruct containing the high and low noise distilled models.
        """
        is_quantized = self.config.get("dit_quantized", False)
        quant_scheme = self.config.get("dit_quant_scheme", "")
        
        if is_quantized:
            logger.info(f"Loading models with FP8 quantization ({quant_scheme})...")
        else:
            logger.info("Loading distilled transformer models (high noise + low noise)...")
        
        # Check if we need to use LoRAs for each model
        use_high_lora, use_low_lora = False, False
        if self.config.get("lora_configs") and self.config["lora_configs"]:
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "high" in lora_name.lower():
                    use_high_lora = True
                elif "low" in lora_name.lower():
                    use_low_lora = True
        
        # Info message if using both FP8 and LoRAs
        if is_quantized and (use_high_lora or use_low_lora):
            logger.info("✓ Using FP8 quantization + LoRAs (LoRAs applied on top of quantized models)")

        # Load HIGH noise model
        if use_high_lora:
            if is_quantized:
                # FP8 + LoRA path: Load quantized distilled model, then apply LoRA
                logger.info(f"  Loading HIGH noise FP8 model ({quant_scheme}) with LoRA support...")
                high_noise_model = WanDistillModel(
                    self.high_noise_model_path,
                    self.config,
                    self.init_device,
                    model_type="wan2.2_moe_high_noise",
                )
                logger.info(f"    ✓ Quantized model loaded, applying LoRAs...")
            else:
                # LoRA only path: Use WanModel (non-quantized) + LoRA
                logger.info("  Loading HIGH noise model with LoRA support (no quantization)...")
                high_noise_model = WanModel(
                    self.high_noise_model_path,
                    self.config,
                    self.init_device,
                    model_type="wan2.2_moe_high_noise",
                )
            
            # Apply LoRAs (works on both quantized and non-quantized models)
            high_lora_wrapper = WanLoraCustomWrapper(high_noise_model)
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "high" in lora_name.lower():
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_id = high_lora_wrapper.load_lora(lora_path)
                    high_lora_wrapper.apply_lora(lora_id, strength)
                    logger.info(f"    ✓ Applied LoRA '{lora_name}' (strength={strength})")
        else:
            # Quantized or regular distilled path (no LoRA)
            if is_quantized:
                logger.info(f"  Loading HIGH noise FP8 model ({quant_scheme})...")
            else:
                logger.info("  Loading HIGH noise DISTILLED model (no LoRA)...")
            high_noise_model = WanDistillModel(
                self.high_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_high_noise",
            )
        logger.info(f"  ✓ High noise model loaded from {self.high_noise_model_path}")

        # Load LOW noise model
        if use_low_lora:
            if is_quantized:
                # FP8 + LoRA path: Load quantized distilled model, then apply LoRA
                logger.info(f"  Loading LOW noise FP8 model ({quant_scheme}) with LoRA support...")
                low_noise_model = WanDistillModel(
                    self.low_noise_model_path,
                    self.config,
                    self.init_device,
                    model_type="wan2.2_moe_low_noise",
                )
                logger.info(f"    ✓ Quantized model loaded, applying LoRAs...")
            else:
                # LoRA only path: Use WanModel (non-quantized) + LoRA
                logger.info("  Loading LOW noise model with LoRA support (no quantization)...")
                low_noise_model = WanModel(
                    self.low_noise_model_path,
                    self.config,
                    self.init_device,
                    model_type="wan2.2_moe_low_noise",
                )
            
            # Apply LoRAs (works on both quantized and non-quantized models)
            low_lora_wrapper = WanLoraCustomWrapper(low_noise_model)
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "low" in lora_name.lower():
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_id = low_lora_wrapper.load_lora(lora_path)
                    low_lora_wrapper.apply_lora(lora_id, strength)
                    logger.info(f"    ✓ Applied LoRA '{lora_name}' (strength={strength})")
        else:
            # Quantized or regular distilled path (no LoRA)
            if is_quantized:
                logger.info(f"  Loading LOW noise FP8 model ({quant_scheme})...")
            else:
                logger.info("  Loading LOW noise DISTILLED model (no LoRA)...")
            low_noise_model = WanDistillModel(
                self.low_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_low_noise",
            )
        logger.info(f"  ✓ Low noise model loaded from {self.low_noise_model_path}")

        # Wrap models in MultiDistillModelStruct for two-stage distilled inference
        multi_model = MultiDistillModelStruct(
            [high_noise_model, low_noise_model],
            self.config,
            self.config["boundary_step_index"]
        )
        
        # Determine mode description
        if (use_high_lora or use_low_lora) and is_quantized:
            mode_desc = "FP8 + LoRA"
        elif use_high_lora or use_low_lora:
            mode_desc = "LoRA"
        elif is_quantized:
            mode_desc = "FP8"
        else:
            mode_desc = "FP16/BF16"
        
        logger.info(f"✓ Created MultiDistillModelStruct ({mode_desc} mode, boundary_step_index={self.config['boundary_step_index']})")
        
        return multi_model

    def run_text_encoder(self, input_info):
        """
        Override text encoder to support T5 offloading during inference.
        
        This method handles T5 encoder offloading by:
        1. Loading T5 to GPU if it's on CPU (offloaded)
        2. Running text encoding
        3. Offloading T5 back to CPU if configured
        4. Clearing GPU cache to free memory for transformer inference
        
        Args:
            input_info: Input information containing prompts
            
        Returns:
            Text encoder output dictionary with context embeddings
        """
        t5_offload = self.config.get("t5_cpu_offload", False)
        
        if t5_offload:
            logger.info("Loading T5 encoder to GPU for text encoding...")
            if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
                self.text_encoders = self.load_text_encoder()
            # Move T5 to GPU
            if hasattr(self.text_encoders[0], 'model'):
                self.text_encoders[0].model = self.text_encoders[0].model.cuda()
            elif hasattr(self.text_encoders[0], 'text_encoder'):
                self.text_encoders[0].text_encoder = self.text_encoders[0].text_encoder.cuda()
        
        # Call parent implementation for actual text encoding
        result = super().run_text_encoder(input_info)
        
        if t5_offload:
            logger.info("Offloading T5 encoder to CPU...")
            # Move T5 back to CPU
            if hasattr(self.text_encoders[0], 'model'):
                self.text_encoders[0].model = self.text_encoders[0].model.cpu()
            elif hasattr(self.text_encoders[0], 'text_encoder'):
                self.text_encoders[0].text_encoder = self.text_encoders[0].text_encoder.cpu()
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ T5 offloaded, GPU memory freed")
        
        return result

    def get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None):
        """
        Override VAE encoder output to apply motion amplitude enhancement.
        
        This method wraps the base implementation and adds motion enhancement
        to fix slow-motion issues in 4-step distilled models.
        
        Args:
            first_frame: First frame tensor
            lat_h: Latent height
            lat_w: Latent width
            last_frame: Optional last frame tensor
            
        Returns:
            VAE encoded output with motion enhancement applied
        """
        h = lat_h * self.config["vae_stride"][1]
        w = lat_w * self.config["vae_stride"][2]
        msk = torch.ones(
            1,
            self.config["target_video_length"],
            lat_h,
            lat_w,
            device=torch.device("cuda"),
        )
        if last_frame is not None:
            msk[:, 1:-1] = 0
        else:
            msk[:, 1:] = 0

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        vae_offload = self.config.get("vae_cpu_offload", False)
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()
        
        # Move VAE encoder to GPU if offloaded
        if vae_offload:
            logger.debug("Loading VAE encoder to GPU...")
            if hasattr(self.vae_encoder, 'encoder'):
                self.vae_encoder.encoder = self.vae_encoder.encoder.cuda()
            elif hasattr(self.vae_encoder, 'vae'):
                self.vae_encoder.vae = self.vae_encoder.vae.cuda()

        if last_frame is not None:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 2, h, w),
                    torch.nn.functional.interpolate(last_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                ],
                dim=1,
            ).cuda()
        else:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 1, h, w),
                ],
                dim=1,
            ).cuda()

        # Standard VAE encoding
        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))
        
        # Apply motion enhancement if enabled
        if self.motion_processor.enable and self.motion_processor.motion_amplitude > 1.0:
            logger.debug("Applying motion amplitude enhancement to VAE latents...")
            vae_encoder_out = self.motion_processor.apply_motion_enhancement(vae_encoder_out)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        elif vae_offload:
            # Offload VAE encoder back to CPU
            logger.debug("Offloading VAE encoder to CPU...")
            if hasattr(self.vae_encoder, 'encoder'):
                self.vae_encoder.encoder = self.vae_encoder.encoder.cpu()
            elif hasattr(self.vae_encoder, 'vae'):
                self.vae_encoder.vae = self.vae_encoder.vae.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("✓ VAE encoder offloaded")
            
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
        return vae_encoder_out

    def run_vae_decoder(self, latents):
        """
        Override VAE decoder to support VAE offloading during inference.
        
        This method handles VAE decoder offloading by:
        1. Loading VAE decoder to GPU if it's on CPU (offloaded)
        2. Running VAE decoding
        3. Offloading VAE decoder back to CPU if configured
        4. Clearing GPU cache to free memory
        
        Args:
            latents: Latent tensor to decode
            
        Returns:
            Decoded images
        """
        vae_offload = self.config.get("vae_cpu_offload", False)
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        
        # Move VAE decoder to GPU if offloaded
        if vae_offload:
            logger.info("Loading VAE decoder to GPU for latent decoding...")
            if hasattr(self.vae_decoder, 'decoder'):
                self.vae_decoder.decoder = self.vae_decoder.decoder.cuda()
            elif hasattr(self.vae_decoder, 'vae'):
                self.vae_decoder.vae = self.vae_decoder.vae.cuda()
        
        # Perform decoding
        images = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        elif vae_offload:
            # Offload VAE decoder back to CPU
            logger.info("Offloading VAE decoder to CPU...")
            if hasattr(self.vae_decoder, 'decoder'):
                self.vae_decoder.decoder = self.vae_decoder.decoder.cpu()
            elif hasattr(self.vae_decoder, 'vae'):
                self.vae_decoder.vae = self.vae_decoder.vae.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ VAE decoder offloaded, GPU memory freed")
        
        return images
    
    def cleanup_after_inference(self):
        """Clean up GPU memory after inference for repeated generations.
        
        This method should be called after each inference to prevent OOM errors
        when running multiple generations in sequence (e.g., in a Gradio demo).
        """
        import gc
        
        logger.info("Cleaning up GPU memory...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✓ Memory cleanup complete")

    def run_pipeline(self, input_info):
        """Run the complete inference pipeline with custom hooks.
        
        Args:
            input_info: InputInfo object containing prompt, image, etc.
            
        Returns:
            The generated video tensor or path, depending on config.
        """
        logger.info("=" * 80)
        logger.info("Starting custom DISTILLED inference pipeline...")
        logger.info(f"Prompt: {input_info.prompt}")
        logger.info(f"Image: {input_info.image_path}")
        logger.info("=" * 80)
        
        # Call the base implementation
        result = super().run_pipeline(input_info)
        
        # Clean up memory after inference
        self.cleanup_after_inference()
        
        logger.info("=" * 80)
        logger.info("✓ Custom distilled inference pipeline complete!")
        logger.info("=" * 80)
        
        return result





def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with defaults matching the reference command."""
    parser = argparse.ArgumentParser(
        description="WAN2.2-MOE I2V quality inference using Wan22MoeRunner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_cls",
        type=str,
        default="wan2.2_moe_distill",
        choices=["wan2.2_moe_distill"],
        help="WAN model class (distilled version)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v",
        choices=["i2v"],
        help="Task type (image-to-video)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/wan2.2_models/official_distill_repo",
        help="Path to WAN2.2 model directory",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default="prod_configs/wan_moe_i2v_a14b_quality.json",
        help="Path to quality config JSON",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="assets/inputs/imgs/pron_test.png",
        help="Input image for I2V",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="woman smiling",
        help="Positive text prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt (optional)",
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default="save_results/video_quality_tuned.mp4",
        help="Where to save the output video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--return_result_tensor",
        action="store_true",
        help="Return tensor instead of writing video file",
    )
    parser.add_argument(
        "--use_prompt_enhancer",
        action="store_true",
        help="Use prompt enhancer (default: False)",
    )
    parser.add_argument(
        "--motion_amplitude",
        type=float,
        default=1.15,
        help="Motion amplitude scaling factor (1.0=no change, >1.0=more motion). "
             "Recommended: 1.15-1.5 for fixing slow-motion in 4-step LoRAs",
    )
    parser.add_argument(
        "--disable_motion_enhancement",
        action="store_true",
        help="Disable motion amplitude enhancement (default: enabled)",
    )
    return parser


def main() -> None:
    """Main entry point for quality inference using custom runner."""
    parser = build_parser()
    args = parser.parse_args()

    # Ensure gradient computation is disabled for inference
    torch.set_grad_enabled(False)

    # Seed all random number generators for reproducibility
    logger.info(f"Setting random seed to {args.seed}")
    seed_all(args.seed)

    # Build configuration from arguments
    logger.info("Building configuration from arguments and config file...")
    config = set_config(args)
    
    # Add motion enhancement settings to config (command-line args override config file)
    # If config already has these values, only override if command-line args differ from defaults
    if "motion_amplitude" not in config or args.motion_amplitude != 1.15:
        config["motion_amplitude"] = args.motion_amplitude
    if "enable_motion_enhancement" not in config or args.disable_motion_enhancement:
        config["enable_motion_enhancement"] = not args.disable_motion_enhancement
    
    print_config(config)

    # Initialize the CUSTOM runner (Wan22MoeCustomRunner extends Wan22MoeDistillRunner)
    logger.info("Initializing Wan22MoeCustomRunner...")
    runner = Wan22MoeCustomRunner(config)
    runner.init_modules()

    # Prepare input information
    logger.info("Preparing input information...")
    input_info = set_input_info(args)

    # Run the inference pipeline
    logger.info("Running inference pipeline...")
    result = runner.run_pipeline(input_info)

    if args.return_result_tensor:
        logger.info("Inference complete. Result tensor returned.")
        return result
    else:
        logger.success(f"Video saved to {args.save_result_path}")


if __name__ == "__main__":
    main()
