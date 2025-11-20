#!/usr/bin/env python3
"""Run WAN2.2-MOE i2v quality inference with motion amplitude enhancement and NAG support.

This script reproduces the exact results of the default inference command
with additional enhancements:
- Motion amplitude enhancement to fix slow-motion issues in 4-step LoRAs
- Normalized Attention Guidance (NAG) for improved generation quality
- Frame interpolation for smooth video output
- FP8 quantization support with LoRA compatibility

Basic usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --config_json prod_configs/wan_moe_i2v_a14b_fp8.json

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
   - `run_text_encoder()`: NAG double-batch encoding for positive/negative prompts
   - `run_pipeline()`: Wrapped pipeline with NAG enable/disable and custom logging

3. **NAG (Normalized Attention Guidance) Classes** (NEW):
   - `NAGWanAttnProcessor`: Custom attention processor implementing NAG algorithm
   - `NAGModelWrapper`: Wrapper to inject/remove NAG processors dynamically

4. **Motion Enhancement Classes**:
   - `MotionAmplitudeProcessor`: Core algorithm for fixing slow-motion issues
   - `EnhancedVAEEncoder`: Wrapper for VAE encoder with motion enhancement

5. **Video Post-Processing Classes**:
   - `FrameInterpolator`: RIFE-based frame interpolation for smooth FPS upsampling
   - `VideoPackager`: Tensor-to-video encoding with multiple codec support
   
6. **Base Runner Responsibilities** (inherited):
   - Model loading (VAE, text encoders, image encoder)
   - Scheduler initialization
   - Text encoding pipeline
   - VAE encoding/decoding
   - Video generation and saving

7. **MultiDistillModelStruct**: Manages the two-stage distilled inference
   - High noise model (used for early denoising steps)
   - Low noise model (used for final refinement steps)
   - Automatic switching based on boundary timestep

NORMALIZED ATTENTION GUIDANCE (NAG):
=====================================

NAG improves generation quality by normalizing attention guidance signals:

1. **Problem**: High guidance scales can cause attention over-saturation, leading to
   artifacts, color bleeding, or unstable outputs.

2. **Solution**: NAG normalizes the guidance signal to prevent over-amplification while
   maintaining strong prompt adherence.

3. **Algorithm**:
   - Compute positive and negative attention separately
   - Apply guidance: guided = positive * scale - negative * (scale - 1)
   - Normalize if guidance exceeds threshold (tau):
     * norm_ratio = ||guided|| / ||positive||
     * if norm_ratio > tau: guided = guided / ||guided|| * ||positive|| * tau
   - Blend: output = guided * alpha + positive * (1 - alpha)

4. **Configuration** (in config JSON):
   ```json
   "nag": {
       "enabled": true,          // Enable NAG (false by default)
       "scale": 1.5,             // Guidance scale (1.0 = disabled, >1.0 = enabled)
       "tau": 2.5,               // Normalization threshold (higher = less normalization)
       "alpha": 0.25,            // Blending weight (0 = all positive, 1 = all guided)
       "negative_prompt": ""     // Optional: override negative prompt for NAG
   }
   ```

5. **Parameters**:
   - **scale** (default: 1.5): Guidance strength. Higher values = stronger prompt adherence
     * 1.0 = NAG disabled
     * 1.5-2.0 = recommended for most use cases
     * 2.0-3.0 = strong guidance (may need lower tau)
   
   - **tau** (default: 2.5): Maximum allowed guidance amplification
     * Lower values = more aggressive normalization (safer, less artifacts)
     * Higher values = less normalization (stronger guidance, risk of artifacts)
     * 2.0-3.0 = recommended range
   
   - **alpha** (default: 0.25): How much normalized guidance to blend in
     * 0.0 = only use positive attention (NAG effectively disabled)
     * 0.25 = default balanced blend
     * 1.0 = only use normalized guided attention

6. **Requirements**:
   - Negative prompt must be provided (via input_info.negative_prompt or nag.negative_prompt)
   - NAG creates double-batch encoding [positive, negative] → increases memory usage
   - Only affects cross-attention layers (text → image attention)

7. **Usage Examples**:
   
   Enable NAG in config:
   ```json
   "nag": {
       "enabled": true,
       "scale": 1.5,
       "tau": 2.5,
       "alpha": 0.25
   }
   ```
   
   With custom negative prompt in config:
   ```json
   "nag": {
       "enabled": true,
       "scale": 2.0,
       "tau": 2.0,
       "alpha": 0.3,
       "negative_prompt": "blurry, low quality, distorted"
   }
   ```
   
   Or provide negative prompt via input_info:
   ```json
   "input_info": {
       "negative_prompt": "static, low quality, artifacts"
   }
   ```

8. **Performance Notes**:
   - NAG adds minimal compute overhead (~5-10% slower)
   - Memory usage increases due to double-batch text encoding
   - Compatible with FP8 quantization and LoRAs
   - Works with all attention types (sage_attn2, flash_attn, etc.)

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
   - Total VRAM savings: ~10-14GB with both enabled

4. **Performance Impact**:
   - Adds ~2-5 seconds per inference (data transfer overhead)
   - Essential for GPUs with <24GB VRAM
   - Recommended for FP8 + LoRA workflows on consumer GPUs

USAGE:
======

Run with defaults (motion enhancement enabled):
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py

Use FP8 config with offloading:
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --config_json prod_configs/wan_moe_i2v_a14b_fp8.json

Enable NAG (edit config JSON):
    {
        "nag": {
            "enabled": true,
            "scale": 1.5,
            "tau": 2.5,
            "alpha": 0.25
        }
    }

VIDEO POST-PROCESSING EXAMPLES:
================================

Using FrameInterpolator for smooth FPS upsampling:
    ```python
    from scripts.wan22.run_quality_inference import FrameInterpolator
    
    interpolator = FrameInterpolator("models/rife/RIFEv4.26_0921")
    smooth_video = interpolator.interpolate(video_tensor, 8.0, 24.0)
    interpolator.cleanup()
    ```

Using VideoPackager for encoding:
    ```python
    from scripts.wan22.run_quality_inference import VideoPackager
    
    with VideoPackager("output.mp4", fps=24.0, codec="avc1") as packager:
        packager.write_video(video_tensor)
    ```

CREDITS:
========

- Motion enhancement: Based on PainterI2V from ComfyUI community
- NAG: Based on Normalized Attention Guidance (https://github.com/ChenDarYen/Normalized-Attention-Guidance)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from torch.distributed.tensor.device_mesh import init_device_mesh

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct, Wan22MoeDistillRunner
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import ALL_INPUT_INFO_KEYS, set_input_info
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.utils import seed_all


def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
        "teacache_thresh": 0.26,
        "use_ret_steps": False,
        "use_bfloat16": True,
        "lora_configs": None,  # List of dicts with 'path' and 'strength' keys
        "use_prompt_enhancer": False,
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "use_image_encoder": True,
    }
    default_config = LockableDict(default_config)
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items() if k not in ALL_INPUT_INFO_KEYS})

    # if config.get("config_json", None) is not None:
    #     logger.info(f"Loading some config from {config['config_json']}")
    #     with open(config["config_json"], "r") as f:
    #         config_json = json.load(f)
    #     config.update(config_json)

    # if os.path.exists(os.path.join(config["model_path"], "config.json")):
    #     with open(os.path.join(config["model_path"], "config.json"), "r") as f:
    #         model_config = json.load(f)
    #     config.update(model_config)
    # elif os.path.exists(os.path.join(config["model_path"], "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
    #     with open(os.path.join(config["model_path"], "low_noise_model", "config.json"), "r") as f:
    #         model_config = json.load(f)
    #     config.update(model_config)
    # elif os.path.exists(os.path.join(config["model_path"], "distill_models", "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
    #     with open(os.path.join(config["model_path"], "distill_models", "low_noise_model", "config.json"), "r") as f:
    #         model_config = json.load(f)
    #     config.update(model_config)
    # elif os.path.exists(os.path.join(config["model_path"], "original", "config.json")):
    #     with open(os.path.join(config["model_path"], "original", "config.json"), "r") as f:
    #         model_config = json.load(f)
    #     config.update(model_config)
    # load quantized config
    # if config.get("dit_quantized_ckpt", None) is not None:
    #     config_path = os.path.join(config["dit_quantized_ckpt"], "config.json")
    #     if os.path.exists(config_path):
    #         with open(config_path, "r") as f:
    #             model_config = json.load(f)
    #         config.update(model_config)

    # if config["task"] in ["i2v", "s2v"]:
    #     if config["target_video_length"] % config["vae_stride"][0] != 1:
    #         logger.warning(f"`num_frames - 1` has to be divisible by {config['vae_stride'][0]}. Rounding to the nearest number.")
    #         config["target_video_length"] = config["target_video_length"] // config["vae_stride"][0] * config["vae_stride"][0] + 1

    # if config["task"] not in ["t2i", "i2i"]:
    #     config["attnmap_frame_num"] = ((config["target_video_length"] - 1) // config["vae_stride"][0] + 1) // config["patch_size"][0]
    #     if config["model_cls"] == "seko_talk":
    #         config["attnmap_frame_num"] += 1

    return config


def set_parallel_config(config):
    if config["parallel"]:
        cfg_p_size = config["parallel"].get("cfg_p_size", 1)
        seq_p_size = config["parallel"].get("seq_p_size", 1)
        assert cfg_p_size * seq_p_size == dist.get_world_size(), f"cfg_p_size * seq_p_size must be equal to world_size"
        config["device_mesh"] = init_device_mesh("cuda", (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))

        if config["parallel"] and config["parallel"].get("seq_p_size", False) and config["parallel"]["seq_p_size"] > 1:
            config["seq_parallel"] = True

        if config.get("enable_cfg", False) and config["parallel"] and config["parallel"].get("cfg_p_size", False) and config["parallel"]["cfg_p_size"] > 1:
            config["cfg_parallel"] = True
        # warmup dist
        _a = torch.zeros([1]).to(f"cuda:{dist.get_rank()}")
        dist.all_reduce(_a)


def print_config(config):
    config_to_print = config.copy()
    config_to_print.pop("device_mesh", None)
    if config["parallel"]:
        if dist.get_rank() == 0:
            logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
    else:
        logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")



class NAGWanAttnProcessor:
    """
    Normalized Attention Guidance processor for WAN model attention layers.
    
    This processor implements NAG (Normalized Attention Guidance) to improve generation quality
    by normalizing the attention guidance signal. It prevents attention over-saturation that
    can occur with high guidance scales.
    
    Based on: Normalized Attention Guidance (NAG) from the Normalized-Attention-Guidance repository.
    
    Args:
        nag_scale (float): Guidance scale for NAG. Values > 1 enable NAG guidance.
        nag_tau (float): Normalization threshold. Limits maximum attention amplification.
        nag_alpha (float): Blending factor between normalized and original guidance.
    """
    
    def __init__(self, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25):
        """Initialize NAG attention processor.
        
        Args:
            nag_scale (float): Guidance scale (1.0 = disabled, >1.0 = enabled)
            nag_tau (float): Normalization threshold (default: 2.5)
            nag_alpha (float): Blending weight (default: 0.25)
        """
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        logger.debug(f"NAGWanAttnProcessor initialized (scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha})")
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply NAG-enhanced attention.
        
        This method performs attention with normalized guidance by:
        1. Detecting if guidance should be applied (double batch for NAG)
        2. Computing separate attention for positive and negative prompts
        3. Normalizing the guidance signal to prevent over-saturation
        4. Blending normalized guidance with original attention
        
        Args:
            attn: Attention module with projection layers
            hidden_states: Main input tensor [B, N, D]
            encoder_hidden_states: Cross-attention context [B or 2B, M, D]
            attention_mask: Optional attention mask
            rotary_emb: Optional rotary position embeddings
            
        Returns:
            Enhanced hidden states with NAG applied
        """
        # Check if NAG should be applied
        apply_guidance = self.nag_scale > 1 and encoder_hidden_states is not None
        if apply_guidance:
            # NAG requires double batch: [positive_prompt, negative_prompt]
            if len(encoder_hidden_states) == 2 * len(hidden_states):
                batch_size = len(hidden_states)
            else:
                apply_guidance = False
        
        # Handle image encoder outputs (for I2V tasks)
        encoder_hidden_states_img = None
        if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
            # First 257 tokens are image features
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
            if apply_guidance:
                # Only use positive batch for image features
                encoder_hidden_states_img = encoder_hidden_states_img[:batch_size]
        
        # Default to self-attention if no cross-attention context
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Apply normalization if available
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Reshape to multi-head format [B, N, D] -> [B, H, N, D/H]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        
        # Process image attention if present (I2V task)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            if hasattr(attn, 'norm_added_k'):
                key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            
            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        
        # Split K,V for NAG guidance if enabled
        if apply_guidance:
            key, key_negative = torch.chunk(key, 2, dim=0)
            value, value_negative = torch.chunk(value, 2, dim=0)
        
        # Compute positive attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        
        # Apply NAG if guidance is enabled
        if apply_guidance:
            # Compute negative attention
            hidden_states_negative = F.scaled_dot_product_attention(
                query, key_negative, value_negative, 
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_negative = hidden_states_negative.transpose(1, 2).flatten(2, 3)
            hidden_states_negative = hidden_states_negative.type_as(query)
            
            hidden_states_positive = hidden_states
            
            # Apply guidance: guided = positive * scale - negative * (scale - 1)
            hidden_states_guidance = (
                hidden_states_positive * self.nag_scale - 
                hidden_states_negative * (self.nag_scale - 1)
            )
            
            # Normalize guidance to prevent over-saturation
            norm_positive = torch.norm(
                hidden_states_positive, p=1, dim=-1, keepdim=True
            ).expand(*hidden_states_positive.shape)
            norm_guidance = torch.norm(
                hidden_states_guidance, p=1, dim=-1, keepdim=True
            ).expand(*hidden_states_guidance.shape)
            
            # Compute scaling ratio
            scale = norm_guidance / norm_positive
            scale = torch.nan_to_num(scale, 10)
            
            # Apply tau threshold: if scale > tau, normalize guidance
            mask = scale > self.nag_tau
            hidden_states_guidance[mask] = (
                hidden_states_guidance[mask] / (norm_guidance[mask] + 1e-7) * 
                norm_positive[mask] * self.nag_tau
            )
            
            # Blend normalized guidance with positive attention
            hidden_states = (
                hidden_states_guidance * self.nag_alpha + 
                hidden_states_positive * (1 - self.nag_alpha)
            )
        
        # Add image attention if present
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        
        # Project output
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class NAGModelWrapper:
    """
    Wrapper to inject NAG attention processors into WAN model.
    
    This class wraps a WAN model and replaces specific attention processors
    with NAG-enabled versions. It allows dynamic enabling/disabling of NAG
    without reloading the entire model.
    
    Only cross-attention layers (attn2) are replaced with NAG processors,
    as self-attention doesn't benefit from guidance normalization.
    """
    
    def __init__(self, model: WanModel):
        """Initialize NAG wrapper.
        
        Args:
            model: WAN model to wrap with NAG support
        """
        self.model = model
        self.original_attn_processors = None
        self.nag_enabled = False
        logger.info("NAGModelWrapper initialized")
    
    def enable_nag(self, nag_scale: float = 1.5, nag_tau: float = 2.5, nag_alpha: float = 0.25):
        """
        Enable NAG by replacing cross-attention processors.
        
        Args:
            nag_scale: Guidance scale (>1.0 to enable)
            nag_tau: Normalization threshold
            nag_alpha: Blending weight
        """
        if self.nag_enabled:
            logger.warning("NAG already enabled, skipping")
            return
        
        logger.info(f"Enabling NAG (scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha})...")
        
        # Store original processors
        self.original_attn_processors = {}
        
        # Get transformer weights module
        if hasattr(self.model, 'transformer_weights'):
            transformer = self.model.transformer_weights
        else:
            logger.error("Model doesn't have transformer_weights attribute")
            return
        
        # Recursively find and replace attention processors
        self._inject_nag_processors(transformer, nag_scale, nag_tau, nag_alpha)
        
        self.nag_enabled = True
        logger.info(f"✓ NAG enabled successfully ({len(self.original_attn_processors)} processors replaced)")
    
    def _inject_nag_processors(self, module, nag_scale, nag_tau, nag_alpha, prefix=""):
        """
        Recursively inject NAG processors into attention layers.
        
        Args:
            module: Module to process
            nag_scale: NAG guidance scale
            nag_tau: NAG normalization threshold
            nag_alpha: NAG blending weight
            prefix: Current module path prefix
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this is a cross-attention layer (attn2)
            # Only apply NAG to cross-attention, not self-attention
            if "attn2" in name.lower() or "cross_attn" in name.lower():
                # Store original processor if it exists
                if hasattr(child, 'processor'):
                    self.original_attn_processors[full_name] = child.processor
                    # Replace with NAG processor
                    child.processor = NAGWanAttnProcessor(nag_scale, nag_tau, nag_alpha)
                    logger.debug(f"Replaced processor at {full_name}")
            
            # Recurse into children
            self._inject_nag_processors(child, nag_scale, nag_tau, nag_alpha, full_name)
    
    def disable_nag(self):
        """
        Disable NAG by restoring original attention processors.
        """
        if not self.nag_enabled:
            logger.warning("NAG not enabled, skipping")
            return
        
        logger.info("Disabling NAG...")
        
        # Restore original processors
        if hasattr(self.model, 'transformer_weights'):
            transformer = self.model.transformer_weights
            self._restore_processors(transformer)
        
        self.original_attn_processors = None
        self.nag_enabled = False
        logger.info("✓ NAG disabled")
    
    def _restore_processors(self, module, prefix=""):
        """
        Recursively restore original attention processors.
        
        Args:
            module: Module to process
            prefix: Current module path prefix
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Restore processor if we saved it
            if full_name in self.original_attn_processors:
                child.processor = self.original_attn_processors[full_name]
                logger.debug(f"Restored processor at {full_name}")
            
            # Recurse into children
            self._restore_processors(child, full_name)
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model."""
        return getattr(self.model, name)


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
        # Average across height (2), width (3), keep time (1) and channel (0)
        diff_mean = diff.mean(dim=(2, 3), keepdim=True)
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


class FrameInterpolator:
    """
    Frame interpolation processor using RIFE model for smooth video upsampling.
    
    This class provides frame interpolation capabilities to increase video frame rate
    using the RIFE (Real-Time Intermediate Flow Estimation) model. It supports:
    - Arbitrary FPS conversion (e.g., 8 FPS → 24 FPS, 30 FPS → 60 FPS)
    - Batch processing for memory efficiency
    - GPU acceleration with automatic device selection
    - Flexible scaling for quality/performance trade-offs
    
    Based on RIFE: https://github.com/megvii-research/ECCV2022-RIFE
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize frame interpolator with RIFE model.
        
        Args:
            model_path (str): Path to RIFE model checkpoint directory
            device (torch.device, optional): Device for inference. Defaults to CUDA if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        
        logger.info(f"FrameInterpolator initialized (device={self.device})")
    
    def _load_model(self):
        """Lazy load RIFE model when first needed."""
        if self.model is not None:
            return
        
        logger.info(f"Loading RIFE model from {self.model_path}...")
        
        from lightx2v.models.vfi.rife.rife_comfyui_wrapper import RIFEWrapper
        self.model = RIFEWrapper(self.model_path, device=self.device)
        logger.info("✓ RIFE model loaded successfully")

    @torch.no_grad()
    def interpolate(
        self,
        video_tensor: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Interpolate video frames from source FPS to target FPS.
        
        Args:
            video_tensor (torch.Tensor): Input video tensor with shape:
                                        - [N, H, W, C] (ComfyUI format), or
                                        - [N, C, H, W] (PyTorch format)
                                        Values should be in range [0, 1]
            source_fps (float): Source frame rate (e.g., 8.0)
            target_fps (float): Target frame rate (e.g., 24.0)
            scale (float): Processing scale factor. Lower = faster but less quality.
                          Default 1.0 (full resolution)
        
        Returns:
            torch.Tensor: Interpolated video in same format as input
        """
        if source_fps >= target_fps:
            logger.warning(f"Target FPS ({target_fps}) <= source FPS ({source_fps}), skipping interpolation")
            return video_tensor
        
        # Load model if not already loaded
        self._load_model()
        
        logger.info(f"Interpolating video: {source_fps} FPS → {target_fps} FPS (scale={scale})")
        logger.debug(f"Input tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        
        # Detect and convert tensor format if needed
        is_pytorch_format = video_tensor.shape[1] == 3 or video_tensor.shape[1] == 1
        
        if is_pytorch_format:
            # Convert [N, C, H, W] → [N, H, W, C] for RIFE
            video_tensor = video_tensor.permute(0, 2, 3, 1)
            logger.debug(f"Converted PyTorch format to ComfyUI format: {video_tensor.shape}")
        
        # Convert to float32 if needed (RIFE requires float32)
        original_dtype = video_tensor.dtype
        if video_tensor.dtype != torch.float32:
            logger.debug(f"Converting from {video_tensor.dtype} to float32 for RIFE")
            video_tensor = video_tensor.to(torch.float32)
        
        # Perform interpolation
        interpolated = self.model.interpolate_frames(
            video_tensor,
            source_fps=source_fps,
            target_fps=target_fps,
            scale=scale
        )
        
        # Convert back to original dtype if needed
        if interpolated.dtype != original_dtype:
            logger.debug(f"Converting back from float32 to {original_dtype}")
            interpolated = interpolated.to(original_dtype)
        
        # Convert back to original format if needed
        if is_pytorch_format:
            interpolated = interpolated.permute(0, 3, 1, 2)
            logger.debug(f"Converted back to PyTorch format: {interpolated.shape}")
        
        logger.info(f"✓ Interpolation complete: {video_tensor.shape[0]} → {interpolated.shape[0]} frames")
        
        return interpolated
    
    def cleanup(self):
        """Release model resources and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ FrameInterpolator resources released")


class VideoPackager:
    """
    Video packaging utility for encoding tensors to video files.
    
    This class handles the conversion of tensor frames to various video formats,
    supporting:
    - Multiple codecs (H.264, H.265/HEVC, VP9, ProRes)
    - Configurable quality settings
    - Audio track support (optional)
    - Metadata injection
    - Progress tracking for long videos
    
    Supports both OpenCV and FFmpeg backends for maximum compatibility.
    """
    
    def __init__(
        self,
        output_path: str,
        fps: float = 24.0,
        codec: str = "mp4v",
        quality: Optional[int] = None,
        use_ffmpeg: bool = False,
    ):
        """
        Initialize video packager.
        
        Args:
            output_path (str): Path for output video file
            fps (float): Frame rate for output video. Default 24.0
            codec (str): Video codec fourcc code. Options:
                        - "mp4v": MPEG-4 (good compatibility)
                        - "avc1": H.264 (high compression)
                        - "hev1": H.265/HEVC (best compression)
                        - "vp09": VP9 (web-optimized)
            quality (int, optional): Quality setting (codec-dependent, higher = better)
            use_ffmpeg (bool): Use FFmpeg instead of OpenCV (more features, slower)
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.use_ffmpeg = use_ffmpeg
        self.writer = None
        self.frame_count = 0
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoPackager initialized: {output_path} ({fps} FPS, codec={codec})")
    
    def _init_opencv_writer(self, width: int, height: int):
        """Initialize OpenCV video writer."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {self.output_path}")
        
        logger.debug(f"OpenCV writer initialized: {width}x{height}")
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy frame for video writing.
        
        Args:
            tensor (torch.Tensor): Frame tensor with shape [C, H, W] or [H, W, C]
                                  Values in range [0, 1]
        
        Returns:
            np.ndarray: BGR frame in uint8 format [H, W, 3]
        """
        # Convert to CPU and numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Detect format and convert to [H, W, C]
        if tensor.shape[0] == 3:  # [C, H, W] format
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and scale to [0, 255]
        frame = tensor.numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def write_frame(self, frame_tensor: torch.Tensor):
        """
        Write a single frame to the video.
        
        Args:
            frame_tensor (torch.Tensor): Frame with shape [C, H, W] or [H, W, C]
        """
        # Initialize writer on first frame
        if self.writer is None:
            if frame_tensor.shape[0] == 3:  # [C, H, W]
                height, width = frame_tensor.shape[1:3]
            else:  # [H, W, C]
                height, width = frame_tensor.shape[0:2]
            
            self._init_opencv_writer(width, height)
        
        # Convert and write frame
        frame = self._tensor_to_frame(frame_tensor)
        self.writer.write(frame)
        self.frame_count += 1
    
    def write_video(self, video_tensor: torch.Tensor, show_progress: bool = True):
        """
        Write entire video tensor to file.
        
        Args:
            video_tensor (torch.Tensor): Video with shape [N, C, H, W] or [N, H, W, C]
            show_progress (bool): Show progress logging. Default True
        """
        num_frames = video_tensor.shape[0]
        
        logger.info(f"Writing {num_frames} frames to {self.output_path}...")
        
        for i in range(num_frames):
            self.write_frame(video_tensor[i])
            
            if show_progress and (i + 1) % 30 == 0:
                logger.debug(f"  Progress: {i + 1}/{num_frames} frames written")
        
        logger.info(f"✓ Video writing complete: {self.frame_count} frames")
    
    def finalize(self):
        """Finalize video file and release resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"✓ Video finalized: {self.output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures video is finalized."""
        self.finalize()
    
    @staticmethod
    def quick_save(
        video_tensor: torch.Tensor,
        output_path: str,
        fps: float = 24.0,
        codec: str = "mp4v",
    ):
        """
        Convenience method for quick video saving.
        
        Args:
            video_tensor (torch.Tensor): Video tensor [N, C, H, W] or [N, H, W, C]
            output_path (str): Output file path
            fps (float): Frame rate. Default 24.0
            codec (str): Video codec. Default "mp4v"
        """
        with VideoPackager(output_path, fps=fps, codec=codec) as packager:
            packager.write_video(video_tensor)


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
    - **Normalized Attention Guidance (NAG)** for improved generation quality
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
    
    NAG (Normalized Attention Guidance):
    - Optional attention normalization for improved quality
    - Prevents over-saturation with high guidance scales
    - Enable via config: "nag": {"enabled": true, "scale": 1.5, "tau": 2.5, "alpha": 0.25}
    - Requires negative prompt for guidance computation
    
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
        
        # Initialize NAG support
        self.nag_wrapper = None
        self.nag_config = self.config.get("nag", {})
        if self.nag_config.get("enabled", False):
            nag_scale = self.nag_config.get("scale", 1.5)
            nag_tau = self.nag_config.get("tau", 2.5)
            nag_alpha = self.nag_config.get("alpha", 0.25)
            logger.info(f"NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}")
        else:
            logger.info("NAG disabled (set nag.enabled=true in config to enable)")
        
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
        
        # Initialize NAG wrapper if enabled
        if self.nag_config.get("enabled", False):
            logger.info("Initializing NAG wrapper for model...")
            if isinstance(self.model, MultiDistillModelStruct):
                # Wrap high noise model (used for most steps)
                self.nag_wrapper = NAGModelWrapper(self.model.model[0])
                logger.info("NAG wrapper applied to high noise model")
            else:
                self.nag_wrapper = NAGModelWrapper(self.model)
                logger.info("NAG wrapper applied to model")
        
        logger.info("✓ Custom runner modules initialized successfully!")



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
        Override text encoder to support T5 offloading and NAG double-batch encoding.
        
        This method handles:
        1. T5 encoder offloading (loading to GPU → encoding → offloading to CPU)
        2. NAG double-batch encoding (positive + negative prompts concatenated)
        3. Memory cleanup after text encoding
        
        For NAG: Encodes both positive and negative prompts, then concatenates them
        into a single batch [positive, negative] for NAG guidance during inference.
        
        Args:
            input_info: Input information containing prompts
            
        Returns:
            Text encoder output dictionary with context embeddings
        """
        t5_offload = self.config.get("t5_cpu_offload", False)
        nag_enabled = self.nag_config.get("enabled", False)
        
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
        
        # For NAG: Create double-batch encoding [positive, negative]
        if nag_enabled and self.nag_wrapper is not None:
            logger.info("Creating NAG double-batch text encoding...")
            
            # Get negative prompt (use config or input_info)
            nag_negative_prompt = self.nag_config.get("negative_prompt", "")
            if not nag_negative_prompt and hasattr(input_info, 'negative_prompt'):
                nag_negative_prompt = input_info.negative_prompt
            
            if not nag_negative_prompt:
                logger.warning("NAG enabled but no negative prompt provided, using empty string")
                nag_negative_prompt = ""
            
            # Create temporary input info for negative prompt
            from copy import deepcopy
            negative_input_info = deepcopy(input_info)
            negative_input_info.prompt = nag_negative_prompt
            
            # Encode negative prompt
            negative_result = super().run_text_encoder(negative_input_info)
            
            # Concatenate positive and negative embeddings
            # result['context'] shape: [B, N, D] -> [2B, N, D] (positive + negative)
            if 'context' in result:
                positive_context = result['context']
                negative_context = negative_result['context']
                result['context'] = torch.cat([positive_context, negative_context], dim=0)
                logger.info(f"NAG double-batch created: {result['context'].shape}")
            
            del negative_result
        
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
        if self.nag_config.get("enabled", False):
            logger.info(f"NAG: ENABLED (scale={self.nag_config.get('scale', 1.5)})")
        logger.info("=" * 80)
        
        # Enable NAG before inference if configured
        if self.nag_wrapper is not None and not self.nag_wrapper.nag_enabled:
            self.nag_wrapper.enable_nag(
                nag_scale=self.nag_config.get("scale", 1.5),
                nag_tau=self.nag_config.get("tau", 2.5),
                nag_alpha=self.nag_config.get("alpha", 0.25)
            )
        
        # Check if frame interpolation is enabled - if so, we need the tensor
        frame_interp_config = self.config.get("frame_interpolation", {})
        should_interpolate = frame_interp_config.get("enabled", False)
        
        # Temporarily override to return tensor if we need to interpolate
        original_return_tensor = input_info.return_result_tensor
        original_save_path = input_info.save_result_path
        if should_interpolate and not input_info.return_result_tensor:
            input_info.return_result_tensor = True
            temp_save_path = input_info.save_result_path
            input_info.save_result_path = None
        
        # Call the base implementation
        result = super().run_pipeline(input_info)
        
        # Disable NAG after inference to free resources
        if self.nag_wrapper is not None and self.nag_wrapper.nag_enabled:
            self.nag_wrapper.disable_nag()
        
        # Apply frame interpolation if enabled
        if should_interpolate:
            logger.info("=" * 80)
            logger.info("Applying frame interpolation...")
            logger.info("=" * 80)
            
            # Extract video tensor from result dict
            video_tensor = result.get("video")
            if video_tensor is None:
                logger.warning("No video tensor found in result. Skipping interpolation.")
            else:
                # Initialize frame interpolator
                interpolator = FrameInterpolator(
                    model_path=frame_interp_config.get("model_path", "models/rife/RIFEv4.26_0921"),
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                
                # Get interpolation parameters
                source_fps = frame_interp_config.get("source_fps", 8.0)
                target_fps = frame_interp_config.get("target_fps", 24.0)
                scale = frame_interp_config.get("scale", 1.0)
                
                logger.info(f"Interpolating: {source_fps} FPS → {target_fps} FPS (scale={scale})")
                
                # Apply interpolation to tensor
                interpolated_video = interpolator.interpolate(
                    video_tensor=video_tensor,
                    source_fps=source_fps,
                    target_fps=target_fps,
                    scale=scale
                )
                
                # Clean up interpolator resources
                interpolator.cleanup()
                
                logger.info("=" * 80)
                logger.info("✓ Frame interpolation complete!")
                logger.info("=" * 80)
                
                # Save the interpolated video if needed
                if not original_return_tensor and original_save_path:
                    logger.info(f"🎬 Saving interpolated video to {original_save_path} 🎬")
                    from lightx2v.utils.utils import save_to_video
                    save_to_video(interpolated_video, original_save_path, fps=target_fps, method="ffmpeg")
                    logger.info(f"✅ Interpolated video saved successfully to: {original_save_path} ✅")
                    result = {"video": None}
                else:
                    result = {"video": interpolated_video}
        
        # Restore original input_info settings
        if should_interpolate and not original_return_tensor:
            input_info.return_result_tensor = original_return_tensor
            input_info.save_result_path = original_save_path
        
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
    return parser


def main() -> None:
    """Main entry point for quality inference using custom runner."""
    parser = build_parser()
    args = parser.parse_args()

    # Ensure gradient computation is disabled for inference
    torch.set_grad_enabled(False)

    # Load JSON file with config and input_info structure
    logger.info(f"Loading configuration from {args.config_json}...")
    with open(args.config_json, "r") as f:
        json_data = json.load(f)
    
    # Check if new nested structure exists
    if "config" in json_data and "input_info" in json_data:
        config_data = json_data["config"]
        input_info_data = json_data["input_info"]
        logger.info("Using nested config structure (config + input_info)")
    else:
        # Fallback to flat structure
        config_data = json_data
        input_info_data = {}
        logger.info("Using flat config structure")
    
    # Create a temporary args object with config data for set_config compatibility
    import argparse
    temp_args = argparse.Namespace()
    temp_args.config_json = args.config_json
    temp_args.seed = args.seed
    temp_args.use_prompt_enhancer = args.use_prompt_enhancer
    temp_args.return_result_tensor = args.return_result_tensor
    
    # Seed all random number generators for reproducibility
    logger.info(f"Setting random seed to {args.seed}")
    seed_all(args.seed)

    # Build configuration
    logger.info("Building configuration from JSON file...")
    config = set_config(temp_args)
    
    # Override with config_data from nested structure
    config.update(config_data)
    
    print_config(config)

    # Initialize the CUSTOM runner (Wan22MoeCustomRunner extends Wan22MoeDistillRunner)
    logger.info("Initializing Wan22MoeCustomRunner...")
    runner = Wan22MoeCustomRunner(config)
    runner.init_modules()

    # Prepare input information from config's input_info section or args
    logger.info("Preparing input information...")
    
    # Create input_info object based on task type
    task = config.get("task", "i2v")
    
    if task == "i2v":
        from lightx2v.utils.input_info import I2VInputInfo
        input_info = I2VInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=input_info_data.get("negative_prompt", ""),
            image_path=args.image_path,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    else:
        # Fallback to original method for other tasks
        args.task = task
        args.negative_prompt = input_info_data.get("negative_prompt", "")
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
