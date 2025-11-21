import argparse
import copy
import json
import os
import random
import subprocess
import sys

"""
Grid search / Random search script for WAN2.2 inference optimization.
"""

BASE_PROMPT = """Sultry beach blowjob at sunset: Blonde beauty in lace bikini kneels, lips parting eagerly for dark man's thick ebony cock, eyes locked in lust. Waves crash; she sucks deep, hands gripping thighs. Warm tones, soft side light, low contrast.
Medium close-up, low-angle handheld pan left-right, zoom on suck, upward tilt. Realistic style.
5s breakdown:
1 (0-1.5s): Approach—cock nears lips, tease; tilt right.
2 (1.5-3.5s): Suck—engulfs, tongue swirl; zoom close, pan left.
3 (3.5-5s): Deep—bob rhythm, pull; surround up to wide reveal."""

NEGATIVE_PROMPT = """bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards, flicker, flickering, jitter, text, text overlays, watermark, logo, heavy blur, motion blur, double faces, bad anatomy, bad proportions, noise, grainy, glitch"""
IMG_LIST = [
    "assets/inputs/imgs/photo-1761755356335.jpeg",
    "assets/inputs/imgs/photo-1761755371106.jpeg",
    "assets/inputs/imgs/photo-1761755380708.jpeg",
]

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
    "target_video_length": [61, 81, 121],
    "scheduler_type": ["EulerScheduler"],
    "nag": [
        {"enabled": False},
        {"enabled": True, "scale": 1.5, "tau": 2.5, "alpha": 0.25},
        {"enabled": True, "scale": 2.0, "tau": 2.5, "alpha": 0.25},
    ],
    # Special key for resolution (height, width) to ensure valid aspect ratios
    "resolution": [
        (720, 1280),
        (480, 832),
        (1024, 1024),
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

def main():
    parser = argparse.ArgumentParser(description="Random search for WAN2.2 inference")
    parser.add_argument("--base_config", type=str, default="prod_configs/wan_moe_i2v_a14b_fp8.json", help="Path to base config JSON")
    parser.add_argument("--output_dir", type=str, default="save_results/force_configs", help="Directory to save results")
    parser.add_argument("--gpu", type=str, default="5", help="GPU ID to use")
    parser.add_argument("--script_path", type=str, default="scripts/wan22/run_quality_inference.py", help="Path to inference script")
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

        # Define output filenames
        video_filename = f"{idx}.mp4"
        json_filename = f"{idx}.json"
        
        video_path = os.path.abspath(os.path.join(args.output_dir, video_filename))
        json_path = os.path.abspath(os.path.join(args.output_dir, json_filename))
        
        # Create a temporary config file for this run
        temp_config_path = os.path.abspath(os.path.join(args.output_dir, f"temp_config_{idx}.json"))
        with open(temp_config_path, "w") as f:
            json.dump(current_config, f, indent=4)

        # Extract input info from config if available to pass as arguments
        input_args = []
        if "input_info" in current_config:
            input_info = current_config["input_info"]
            if "prompt" in input_info:
                input_args.extend(["--prompt", str(input_info["prompt"])])
            if "image_path" in input_info:
                input_args.extend(["--image_path", str(input_info["image_path"])])
            if "seed" in input_info:
                input_args.extend(["--seed", str(input_info["seed"])])

        # Construct the command
        cmd = [
            sys.executable, args.script_path,
            "--config_json", temp_config_path,
            "--save_result_path", video_path
        ] + input_args
        
        # Set environment variables (GPU)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        
        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, env=env, check=True)
            
            # Save the config used
            save_data = {
                "grid_parameters": current_params,
                "full_config": current_config
            }
            with open(json_path, "w") as f:
                json.dump(save_data, f, indent=4)
                
            print(f"✓ Success! Saved video to {video_path} and config to {json_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running sample {idx}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        finally:
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

if __name__ == "__main__":
    main()
