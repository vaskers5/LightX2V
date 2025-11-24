import argparse
import copy
import json
import os
import random
import subprocess
import sys
import torch
import gc
import torch.distributed as dist

# Add script directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightx2v.utils.input_info import I2VInputInfo
from lightx_wan22_moe_custom import (
    set_config, 
    set_input_info, 
    seed_all, 
    set_parallel_config, 
    print_config,
    Wan22MoeFP8CustomRunner,
    Wan22MoeBF16CustomRunner,
    get_default_config
)

"""
Grid search / Random search script for WAN2.2 inference optimization.
"""

BASE_PROMPT = """a  woman in front of a penis, she engages in a deep throat blowjob, she swallows the penis all the way. Her lips touches the man's groin. Her nose smashes against the man's hips."""

NEGATIVE_PROMPT = """bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards, flicker, flickering, jitter, text, text overlays, watermark, logo, heavy blur, motion blur, double faces, bad anatomy, bad proportions, noise, grainy, glitch"""
IMG_LIST = [
    "assets/inputs/imgs/photo-1761755356335.jpeg",
    "assets/inputs/imgs/photo-1761755371106.jpeg",
    "assets/inputs/imgs/photo-1761755380708.jpeg",
]

available_loras = {
    "base_loras":
        [
            [{
                "name": "high_noise_model",
                "path": "models/wan2.2_models/loras/nsfw_lora/high_nsfw_v13_000005000_high_noise_fp8.safetensors",
                "strength": 1.0
            },
            {
                "name": "low_noise_model",
                "path": "models/wan2.2_models/loras/nsfw_lora/low_nsfw_v13_000005000_low_noise_fp8.safetensors",
                "strength": 1.0
            },],
        ],
    "blowjob_loras":[
            [
        {
                "name": "high_noise_model",
                "path": "models/wan2.2_models/loras/bbc_blowjob/wan22-bbcdeepthroat-115epoc-high-k3nk.safetensors",
                "strength": 1.0
            },
            {
                "name": "low_noise_model",
                "path": "models/wan2.2_models/loras/bbc_blowjob/wan22-bbcdeepthroat-155epoc-low-720-k3nk.safetensors",
                "strength": 1.0
            }
        ],
        [
            {
                "name": "high_noise_model",
                "path": "models/wan2.2_models/loras/oral_insertion/wan2.2-i2v-high-oral-insertion-v1.0_fp8.safetensors",
                "strength": 1.0
            },
            {
                "name": "low_noise_model",
                "path": "models/wan2.2_models/loras/oral_insertion/wan2.2-i2v-low-oral-insertion-v1.0_fp8.safetensors",
                "strength": 1.0
            }
        ],
        [
            {
                "name": "high_noise_model",
                "path": "models/wan2.2_models/loras/hand_blow/WAN-2.2-I2V-HandjobBlowjobCombo-HIGH-v1.safetensors",
                "strength": 1.0
            },
            {
                "name": "low_noise_model",
                "path": "models/wan2.2_models/loras/hand_blow/WAN-2.2-I2V-HandjobBlowjobCombo-LOW-v1.safetensors",
                "strength": 1.0
            }
        ],
        [
        {
                "name": "high_noise_model",
                "path": "models/wan2.2_models/loras/ultimate_blow_job/wan22-ultimatedeepthroat-i2v-102epoc-high-k3nk.safetensors",
                "strength": 1.0
        },
        {
            "name": "low_noise_model",
            "path": "models/wan2.2_models/loras/ultimate_blow_job/wan22-ultimatedeepthroat-I2V-101epoc-low-k3nk.safetensors",
            "strength": 1.0
        }
        ],
        [
        {
            "name": "high_noise_model",
            "path": "models/wan2.2_models/loras/ultimate_blow_job/Wan22_ThroatV2_High.safetensors",
            "strength": 1.0
        },
        {
            "name": "low_noise_model",
            "path": "models/wan2.2_models/loras/ultimate_blow_job/Wan22_ThroatV2_Low.safetensors",
            "strength": 1.0
        } 
        ]
    ]
        # [
        # {
        #     "name": "high_noise_model",
        #     "path": "models/wan2.2_models/loras/penis_play/Wan22_ThroatV2_High.safetensors",
        #     "strength": 1.0
        # },
        # {
        #     "name": "low_noise_model",
        #     "path": "models/wan2.2_models/loras/penis_play/Wan22_ThroatV2_Low.safetensors",
        #     "strength": 1.0
        # } 
        # ],
}

available_checkpoints = {
    'fp8':
        {
        "high_noise_quantized_ckpt": "models/wan2.2_models/distill_models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "low_noise_quantized_ckpt": "models/wan2.2_models/distill_models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
        },
    "bf16":
        {
        "high_noise_original_ckpt": "models/wan2.2_models/diffussion_models/wan2.2_i2v_A14b_high_noise_lightx2v_bf16.safetensors",
        "low_noise_original_ckpt": "models/wan2.2_models/diffussion_models/wan2.2_i2v_A14b_low_noise_lightx2v_bf16.safetensors"     
        }
}          

# ==================================================================================
# SEARCH CONFIGURATION
# ==================================================================================
GRID = {
    "sample_shift": [3.0, 4.0, 5.0, 6.0, 7.0],
    "denoising_steps": [
        {"steps": 4, "boundary": 0.5},
        {"steps": 4, "boundary": 0.5},
        {"steps": 4, "boundary": 0.5},
        {"steps": 4, "boundary": 0.5},
        {"steps": 4, "boundary": 0.5},
        {"steps": 6, "boundary": 0.4},
        {"steps": 8, "boundary": 0.5},
        {"steps": 8, "boundary": 0.75},
        {"steps": 10, "boundary": 0.6},
        {"steps": 10, "boundary": 0.8},
        {"steps": 12, "boundary": 0.6},
        {"steps": 12, "boundary": 0.8},

    ],
    "motion_enhancement": [
        {"enabled": False},
        {"enabled": True, "amplitude": 1.0},
        {"enabled": True, "amplitude": 1.1},
        {"enabled": True, "amplitude": 1.15},
        {"enabled": True, "amplitude": 1.2},
        {"enabled": True, "amplitude": 1.3},
    ],
    "target_video_length": [61, 81],
    "scheduler_type": ["EulerScheduler"],
    "nag": [
        {"enabled": False},
        {"enabled": True, "scale": 1.5, "tau": 2.5, "alpha": 0.25, "negative_prompt": NEGATIVE_PROMPT},
        {"enabled": True, "scale": 2.0, "tau": 2.5, "alpha": 0.25, "negative_prompt": NEGATIVE_PROMPT}, 
        {"enabled": True, "scale": 2.5, "tau": 2.5, "alpha": 0.25, "negative_prompt": NEGATIVE_PROMPT},
    ],
    # Special key for resolution (height, width) to ensure valid aspect ratios
    "resolution": [
        (480, 854),
        (720, 1280),
    ],
     "t5_original_ckpt": 
         [
             "models/wan2.2_models/encoders/models_t5_umt5-xxl-enc-bf16.safetensors",
             "models/wan2.2_models/encoders/models_t5_umt5-xxl-enc-bf16_fully_uncensored.safetensors"
             
         ],
    "input_info": [
        {"prompt": BASE_PROMPT, "image_path": IMG_LIST[0], "seed": 42, "negative_prompt": NEGATIVE_PROMPT},
        {"prompt": BASE_PROMPT, "image_path": IMG_LIST[1], "seed": 42, "negative_prompt": NEGATIVE_PROMPT},
        {"prompt": BASE_PROMPT, "image_path": IMG_LIST[2], "seed": 42, "negative_prompt": NEGATIVE_PROMPT},
        {"prompt": BASE_PROMPT, "image_path": IMG_LIST[0], "seed": 42, "negative_prompt": ""},
        {"prompt": BASE_PROMPT, "image_path": IMG_LIST[1], "seed": 42, "negative_prompt": ""},
        {"prompt": BASE_PROMPT, "image_path": IMG_LIST[2], "seed": 42, "negative_prompt": ""},

    ],                    
}
# ==================================================================================

def run_pipeline_direct(mode, config, output_path):
    # Prepare input info
    input_info_dict = config.get("input_info", {})
    
    # Create I2VInputInfo object
    input_info = I2VInputInfo(
        prompt=input_info_dict.get("prompt", ""),
        negative_prompt=input_info_dict.get("negative_prompt", ""),
        image_path=input_info_dict.get("image_path", ""),
        seed=input_info_dict.get("seed", 42),
        save_result_path=output_path,
        return_result_tensor=False
    )
    
    # Set seed
    seed_all(input_info.seed)
    
    # Initialize runner
    if mode == 'fp8':
        runner = Wan22MoeFP8CustomRunner(config)
    else:
        runner = Wan22MoeBF16CustomRunner(config)
        
    runner.init_modules()
    
    # Run pipeline
    runner.run_pipeline(input_info)
    
    # Cleanup
    del runner
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Random search for WAN2.2 inference")
    parser.add_argument("--base_config", type=str, default="prod_configs/wan_moe_i2v_a14b_fp8.json", help="Path to base config JSON")
    parser.add_argument("--output_dir", type=str, default="save_results/fp8_vs_bf16", help="Directory to save results")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of random samples to run")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load base configuration
    if not os.path.exists(args.base_config):
        print(f"Error: Base config file not found at {args.base_config}")
        sys.exit(1)

    print(f"Loading base config from {args.base_config}")
    with open(args.base_config, "r") as f:
        base_config = json.load(f)

    print(f"Starting random search with {args.num_samples} samples.")
    print(f"Output directory: {args.output_dir}")
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    for idx in range(args.num_samples):
        # Sample parameters
        current_params = {}
        for key, options in GRID.items():
            current_params[key] = random.choice(options)
            
        print(f"\n" + "="*60)
        print(f"[{idx+1}/{args.num_samples}] Testing combination: {current_params}")
        print("="*60)
        
        # Deep copy base config
        current_config = copy.deepcopy(base_config)
        
        # Update config with current parameters
        if "config" not in current_config:
            print("Error: Base config does not have a 'config' section.")
            sys.exit(1)

        # Apply parameters
        for k, v in current_params.items():
            if k == "resolution":
                # Unpack resolution tuple
                h, w = v
                current_config["config"]["target_height"] = h
                current_config["config"]["target_width"] = w
            else:
                current_config["config"][k] = v

        # choose loras
        loras_config = []
        if random.randint(0, 3) > 0:
            loras_config.append(random.choice(available_loras["base_loras"]))
        if random.randint(0, 8) > 0:
            for i in range(random.randint(1, 3)):
                loras_config.append(random.choice(available_loras["blowjob_loras"]))
        
        current_config["config"]["lora_configs"] = [item for sublist in loras_config for item in sublist]
        
        # Run for both FP8 and BF16
        for mode in ['fp8', 'bf16']:
            print(f"  Running {mode} pipeline...")
            
            # Create mode-specific config
            mode_config = copy.deepcopy(current_config)
            
            # Update checkpoints based on mode
            if mode in available_checkpoints:
                for k, v in available_checkpoints[mode].items():
                    mode_config["config"][k] = v
            
            # Define output filenames
            video_filename = f"{idx}_{mode}.mp4"
            json_filename = f"{idx}_{mode}.json"
            
            video_path = os.path.abspath(os.path.join(args.output_dir, video_filename))
            json_path = os.path.abspath(os.path.join(args.output_dir, json_filename))
            
            # Save the config used
            save_data = {
                "grid_parameters": current_params,
                "full_config": mode_config,
                "mode": mode
            }
            with open(json_path, "w") as f:
                json.dump(save_data, f, indent=4)

            try:
                # Flatten config logic (replicated from set_config)
                flat_config = get_default_config()
                
                if "config" in mode_config and isinstance(mode_config["config"], dict):
                    flat_config.update(mode_config["config"])
                    for k, v in mode_config.items():
                        if k != "config":
                            flat_config[k] = v
                else:
                    flat_config.update(mode_config)
                
                # Ensure dit_quantized is set for FP8
                if mode == 'fp8' and not flat_config.get("dit_quantized"):
                    flat_config["dit_quantized"] = True
                
                # Set parallel config
                set_parallel_config(flat_config)
                
                print(f"Running {mode} pipeline directly...")
                run_pipeline_direct(mode, flat_config, video_path)
                
                print(f"✓ Success! Saved video to {video_path} and config to {json_path}")
                
            except Exception as e:
                print(f"✗ Error running sample {idx} mode {mode}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
