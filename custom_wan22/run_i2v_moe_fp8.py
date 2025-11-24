"""
truncate -s 0 output.log
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python -u custom_wan22/run_i2v_moe_fp8.py \
  --config_json custom_wan22/default_fp8.json >> output.log 2>&1
"""
#!/usr/bin/env python3
import argparse
import torch
from loguru import logger
import sys
import os
import torch.distributed as dist

# Add script directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightx_wan22_moe_custom import (
    set_config, 
    set_input_info, 
    seed_all, 
    set_parallel_config, 
    print_config,
    Wan22MoeFP8CustomRunner
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WAN2.2-MOE I2V FP8 inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default="prod_configs/wan_moe_i2v_a14b_fp8.json",
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
        default="save_results/updated_video_fp8.mp4",
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
    parser.add_argument(
        "--ram_limit",
        type=float,
        default=110,
        help="RAM limit in GB",
    )
    parser.add_argument(
        "--vram_limit",
        type=float,
        default=64068,
        help="VRAM limit in MB",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Set memory limits
    if args.ram_limit > 0:
        import resource
        try:
            # Convert GB to bytes
            ram_limit_bytes = int(args.ram_limit * 1024 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (ram_limit_bytes, ram_limit_bytes))
            logger.info(f"Set RAM limit to {args.ram_limit} GB")
        except Exception as e:
            logger.warning(f"Failed to set RAM limit: {e}")

    if args.vram_limit > 0 and torch.cuda.is_available():
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
            vram_limit_bytes = int(args.vram_limit * 1024 * 1024)
            if vram_limit_bytes < total_vram:
                fraction = vram_limit_bytes / total_vram
                torch.cuda.set_per_process_memory_fraction(fraction)
                logger.info(f"Set VRAM limit to {args.vram_limit} MB ({fraction:.2%} of total)")
            else:
                logger.warning(f"VRAM limit {args.vram_limit} MB is larger than total VRAM {total_vram/1024/1024:.0f} MB")
        except Exception as e:
            logger.warning(f"Failed to set VRAM limit: {e}")

    torch.set_grad_enabled(False)

    logger.info(f"Loading configuration from {args.config_json}...")
    config = set_config(args)
    
    # Force FP8 if not in config (though runner handles it too)
    if not config.get("dit_quantized"):
        config["dit_quantized"] = True

    set_parallel_config(config)
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print_config(config)
    else:
        print_config(config)

    seed_all(config.get("seed", 42))

    logger.info("Initializing Wan22MoeFP8CustomRunner...")
    runner = Wan22MoeFP8CustomRunner(config)
    runner.init_modules()
    
    args.task = "i2v"
    args.negative_prompt = ""
    input_info = set_input_info(args)

    if "input_info" in config:
        logger.info("Overriding input_info from config...")
        for k, v in config["input_info"].items():
            if hasattr(input_info, k):
                setattr(input_info, k, v)
                logger.info(f"  {k}: {v}")
    
    runner.run_pipeline(input_info)

if __name__ == "__main__":
    main()
