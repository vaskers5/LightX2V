#!/usr/bin/env python3
"""Run WAN2.2-MOE i2v quality inference directly via custom Wan22MoeRunner.

This script reproduces the exact results of the default inference command:

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
   └── Inherits from: **Wan22MoeRunner** (lightx2v.models.runners.wan.wan_runner)
       └── Inherits from: **WanRunner** (base WAN runner class)
           └── Inherits from: **DefaultRunner** (lightx2v.models.runners.default_runner)

2. **Key Customizations**:
   - `__init__()`: Enhanced logging of model paths during initialization
   - `load_transformer()`: Custom LoRA loading using config 'name' field
   - `_apply_loras()`: Helper method for applying LoRAs to high/low noise models
   - `run_pipeline()`: Wrapped pipeline with custom logging hooks

3. **Base Runner Responsibilities** (inherited):
   - Model loading (VAE, text encoders, image encoder)
   - Scheduler initialization
   - Text encoding pipeline
   - VAE encoding/decoding
   - Video generation and saving

4. **MultiModelStruct**: Manages the two-stage distilled inference
   - High noise model (used for early denoising steps)
   - Low noise model (used for final refinement steps)
   - Automatic switching based on boundary timestep

USAGE:
======

Run with defaults (matches the reference command):
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py

Override specific parameters:
    CUDA_VISIBLE_DEVICES=7 python scripts/wan22/run_quality_inference.py \
        --prompt "woman laughing and waving" \
        --save_result_path save_results/custom_output.mp4

Add custom LoRAs (edit the config JSON):
    Add to "lora_configs" in prod_configs/wan_moe_i2v_a14b_quality.json:
    {
        "name": "my_custom_high_noise_lora",
        "path": "models/wan2.2_models/loras/my_lora.safetensors",
        "strength": 0.8
    }
"""

from __future__ import annotations

import argparse

import torch
from loguru import logger

from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct, Wan22MoeDistillRunner
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all


class Wan22MoeCustomRunner(Wan22MoeDistillRunner):
    """Custom WAN2.2-MOE Distill runner with enhanced LoRA handling and extensibility.
    
    This runner extends Wan22MoeDistillRunner (the DISTILLED version) to provide:
    - Custom LoRA loading logic based on config 'name' field
    - Support for distilled models (WanDistillModel)
    - Easy hooks for pre/post-processing customization
    - Better logging and monitoring capabilities
    
    IMPORTANT: This inherits from Wan22MoeDistillRunner, not Wan22MoeRunner,
    because the quality config uses distilled models.
    """

    def __init__(self, config):
        """Initialize the custom runner with the given config.
        
        Args:
            config: Configuration dictionary containing model paths, hyperparameters, etc.
        """
        super().__init__(config)
        logger.info(f"Initialized {self.__class__.__name__} (Distilled version)")
        logger.info(f"High noise model path: {self.high_noise_model_path}")
        logger.info(f"Low noise model path: {self.low_noise_model_path}")

    def load_transformer(self):
        """Load high and low noise distilled transformer models with custom LoRA support.
        
        This method overrides the base implementation to support LoRA loading
        based on the 'name' field in lora_configs. When no LoRAs are used,
        it loads WanDistillModel instead of regular WanModel.
        
        Returns:
            MultiDistillModelStruct containing the high and low noise distilled models.
        """
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

        # Load HIGH noise model (WanModel with LoRA or WanDistillModel without)
        if use_high_lora:
            logger.info("  Loading HIGH noise model with LoRA support...")
            high_noise_model = WanModel(
                self.high_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_high_noise",
            )
            high_lora_wrapper = WanLoraWrapper(high_noise_model)
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "high" in lora_name.lower():
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_id = high_lora_wrapper.load_lora(lora_path)
                    high_lora_wrapper.apply_lora(lora_id, strength)
                    logger.info(f"    ✓ Applied LoRA '{lora_name}' (strength={strength})")
        else:
            logger.info("  Loading HIGH noise DISTILLED model (no LoRA)...")
            high_noise_model = WanDistillModel(
                self.high_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_high_noise",
            )
        logger.info(f"  ✓ High noise model loaded from {self.high_noise_model_path}")

        # Load LOW noise model (WanModel with LoRA or WanDistillModel without)
        if use_low_lora:
            logger.info("  Loading LOW noise model with LoRA support...")
            low_noise_model = WanModel(
                self.low_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_low_noise",
            )
            low_lora_wrapper = WanLoraWrapper(low_noise_model)
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "low" in lora_name.lower():
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_id = low_lora_wrapper.load_lora(lora_path)
                    low_lora_wrapper.apply_lora(lora_id, strength)
                    logger.info(f"    ✓ Applied LoRA '{lora_name}' (strength={strength})")
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
        logger.info(f"✓ Created MultiDistillModelStruct with boundary_step_index={self.config['boundary_step_index']}")
        
        return multi_model

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
