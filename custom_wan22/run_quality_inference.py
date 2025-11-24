#!/usr/bin/env python3
"""Run WAN2.2-MOE i2v quality inference with motion amplitude enhancement and NAG support.

This script reproduces the exact results of the default inference command
with additional enhancements:
- Motion amplitude enhancement to fix slow-motion issues in 4-step LoRAs
- Normalized Attention Guidance (NAG) for improved generation quality
- Frame interpolation for smooth video output
- FP8 quantization support with LoRA compatibility
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import torch
from loguru import logger

# Add script directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightx_wan22_moe_custom import (
    Wan22MoeCustomRunner,
    set_config,
    print_config,
    seed_all,
    set_input_info
)

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
