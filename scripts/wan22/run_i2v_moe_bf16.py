#!/usr/bin/env python3
import argparse
import torch
from loguru import logger
import sys
import os
import torch.distributed as dist

# Add script directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from i2v_moe_custom_base import set_config, set_input_info, seed_all, set_parallel_config, print_config
from i2v_moe_custom_bf16 import Wan22MoeBF16CustomRunner

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WAN2.2-MOE I2V BF16 inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default="prod_configs/wan_moe_i2v_a14b_quality.json",
        help="Path to config JSON",
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
        default="save_results/video_bf16.mp4",
        help="Where to save the output video",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--return_result_tensor",
        action="store_true",
        help="Return tensor instead of writing video file",
    )
    parser.add_argument(
        "--use_prompt_enhancer",
        action="store_true",
        help="Use prompt enhancer",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    logger.info(f"Loading configuration from {args.config_json}...")
    config = set_config(args)
    
    set_parallel_config(config)
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print_config(config)
    else:
        print_config(config)

    seed_all(config.get("seed", 42))

    logger.info("Initializing Wan22MoeBF16CustomRunner...")
    runner = Wan22MoeBF16CustomRunner(config)
    runner.init_modules()
    
    args.task = "i2v"
    args.negative_prompt = ""
    input_info = set_input_info(args)
    
    runner.run_pipeline(input_info)

if __name__ == "__main__":
    main()
