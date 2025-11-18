#!/usr/bin/env python3
"""Gradio demo for WAN2.2-MOE quality inference with full config customization.

This demo allows users to:
- Upload custom images
- Enter custom prompts (positive and negative)
- Adjust all quality settings from the config
- Preview and download generated videos
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import torch
from loguru import logger
from PIL import Image

from scripts.wan22.run_quality_inference import Wan22MoeCustomRunner
from lightx2v.utils.input_info import I2VInputInfo
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import seed_all


class QualityInferenceDemo:
    """Gradio demo for WAN2.2-MOE quality inference."""

    def __init__(self, config_path: str, model_path: str):
        """Initialize the demo with base config.
        
        Args:
            config_path: Path to the base quality config JSON
            model_path: Path to the model directory
        """
        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        
        # Load base configuration
        with open(self.config_path) as f:
            self.base_config = json.load(f)
        
        logger.info(f"Loaded base config from {self.config_path}")
        logger.info(f"Model path: {self.model_path}")
        
        # Runner will be loaded lazily on first inference
        self.runner: Optional[Wan22MoeCustomRunner] = None
        self.current_config_hash: Optional[str] = None

    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Get a hash of config to detect changes that require model reload."""
        # Hash only settings that require model reload
        # NOTE: Video settings, CFG, etc. don't require reload
        important_keys = [
            "cpu_offload",  # Changes offloading behavior
            "lora_configs"  # LoRA changes require reload
        ]
        config_subset = {k: config.get(k) for k in important_keys}
        return str(hash(json.dumps(config_subset, sort_keys=True)))

    def _load_runner(self, config: Dict[str, Any], force_reload: bool = False):
        """Load or reload the runner only if necessary.
        
        Args:
            config: Configuration dictionary
            force_reload: Force reload even if config hasn't changed
        """
        config_hash = self._get_config_hash(config)
        
        if self.runner is None:
            logger.info("Loading runner for the first time...")
            
            # Disable gradients
            torch.set_grad_enabled(False)
            
            # Create runner
            self.runner = Wan22MoeCustomRunner(config)
            self.runner.init_modules()
            self.current_config_hash = config_hash
            
            logger.info("✓ Runner loaded successfully")
        
        elif force_reload or config_hash != self.current_config_hash:
            logger.info("Configuration changed - reloading runner...")
            
            # Clear existing runner
            if self.runner is not None:
                del self.runner
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Disable gradients
            torch.set_grad_enabled(False)
            
            # Create runner
            self.runner = Wan22MoeCustomRunner(config)
            self.runner.init_modules()
            self.current_config_hash = config_hash
            
            logger.info("✓ Runner reloaded successfully")
        else:
            logger.info("✓ Reusing existing runner (config unchanged)")

    def generate_video(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        # Video settings
        video_length: int,
        height: int,
        width: int,
        # Inference settings
        infer_steps: int,
        sample_shift: float,
        cfg_scale_high: float,
        cfg_scale_low: float,
        boundary_step_index: int,
        # LoRA settings
        enable_loras: bool,
        lora_strength: float,
        # Performance settings
        cpu_offload: bool,
        progress=gr.Progress()
    ) -> str:
        """Generate video from image and prompt.
        
        Returns:
            Path to the generated video file
        """
        try:
            progress(0.0, desc="Preparing configuration...")
            
            # Create a modified config
            config = copy.deepcopy(self.base_config)
            config["target_video_length"] = video_length
            config["target_height"] = height
            config["target_width"] = width
            config["infer_steps"] = infer_steps
            config["sample_shift"] = sample_shift
            config["sample_guide_scale"] = [cfg_scale_high, cfg_scale_low]
            config["boundary_step_index"] = boundary_step_index
            config["cpu_offload"] = cpu_offload
            
            # Handle LoRAs
            if not enable_loras:
                config["lora_configs"] = []
            else:
                # Update LoRA strengths
                for lora_config in config.get("lora_configs", []):
                    lora_config["strength"] = lora_strength
            
            # Create temporary args namespace for compatibility
            class Args:
                pass
            
            args = Args()
            args.model_cls = "wan2.2_moe_distill"
            args.task = "i2v"
            args.model_path = str(self.model_path)
            args.config_json = str(self.config_path)
            args.seed = seed
            args.use_prompt_enhancer = False
            args.return_result_tensor = False
            
            progress(0.1, desc="Building configuration...")
            
            # Build full config using set_config
            full_config = set_config(args)
            
            # Merge our custom settings
            full_config.update(config)
            
            progress(0.2, desc="Loading model...")
            
            # Seed RNGs
            seed_all(seed)
            
            # Load runner with current config
            self._load_runner(full_config)
            
            progress(0.3, desc="Preparing input...")
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                image.save(tmp_img.name)
                image_path = tmp_img.name
            
            # Create output path
            output_dir = Path("save_results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"gradio_output_{seed}.mp4"
            
            # Create input info
            input_info = I2VInputInfo(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_path=image_path,
                save_result_path=str(output_path),
                return_result_tensor=False
            )
            
            progress(0.4, desc="Running inference...")
            
            # Run inference
            logger.info(f"Starting inference with prompt: {prompt}")
            self.runner.run_pipeline(input_info)
            
            progress(1.0, desc="Done!")
            
            # Clean up temporary image
            try:
                os.unlink(image_path)
            except Exception:
                pass
            
            logger.info(f"Video saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise gr.Error(f"Inference failed: {str(e)}")


def create_demo(config_path: str, model_path: str) -> gr.Blocks:
    """Create the Gradio demo interface.
    
    Args:
        config_path: Path to base config JSON
        model_path: Path to model directory
        
    Returns:
        Gradio Blocks interface
    """
    demo_instance = QualityInferenceDemo(config_path, model_path)
    
    # Load config for defaults
    with open(config_path) as f:
        default_config = json.load(f)
    
    with gr.Blocks(title="WAN2.2-MOE Quality Inference") as demo:
        gr.Markdown("""
        # 🎬 WAN2.2-MOE Quality Image-to-Video Generation
        
        Transform your images into high-quality videos with customizable settings.
        Upload an image, write a prompt, and adjust the parameters to generate your video.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input")
                
                image_input = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what you want to happen in the video...",
                    value="woman smiling",
                    lines=3
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What you don't want in the video...",
                    value="",
                    lines=2
                )
                
                seed_input = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                    info="Set to -1 for random seed"
                )
                
                gr.Markdown("### 🎥 Video Settings")
                
                video_length_input = gr.Slider(
                    label="Video Length (frames)",
                    minimum=13,
                    maximum=121,
                    step=12,
                    value=default_config.get("target_video_length", 61),
                    info="Number of frames to generate"
                )
                
                with gr.Row():
                    height_input = gr.Slider(
                        label="Height",
                        minimum=480,
                        maximum=1080,
                        step=16,
                        value=default_config.get("target_height", 720)
                    )
                    
                    width_input = gr.Slider(
                        label="Width",
                        minimum=640,
                        maximum=1920,
                        step=16,
                        value=default_config.get("target_width", 1280)
                    )
                
                gr.Markdown("### ⚙️ Inference Settings")
                
                infer_steps_input = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=default_config.get("infer_steps", 4),
                    info="More steps = better quality but slower"
                )
                
                sample_shift_input = gr.Slider(
                    label="Sample Shift",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=default_config.get("sample_shift", 5.0),
                    info="Controls temporal dynamics"
                )
                
                cfg_scale_high_input = gr.Slider(
                    label="CFG Scale (High Noise)",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    value=default_config.get("sample_guide_scale", [1.0, 1.0])[0],
                    info="Guidance strength for early steps"
                )
                
                cfg_scale_low_input = gr.Slider(
                    label="CFG Scale (Low Noise)",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    value=default_config.get("sample_guide_scale", [1.0, 1.0])[1],
                    info="Guidance strength for final steps"
                )
                
                boundary_step_input = gr.Slider(
                    label="Boundary Step Index",
                    minimum=0,
                    maximum=4,
                    step=1,
                    value=default_config.get("boundary_step_index", 2),
                    info="When to switch from high to low noise model"
                )
                
                gr.Markdown("### 🎨 LoRA Settings")
                
                enable_loras_input = gr.Checkbox(
                    label="Enable LoRAs",
                    value=True,
                    info="Use fine-tuned LoRA models"
                )
                
                lora_strength_input = gr.Slider(
                    label="LoRA Strength",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    info="Strength of LoRA effect"
                )
                
                gr.Markdown("### 💻 Performance")
                
                cpu_offload_input = gr.Checkbox(
                    label="CPU Offload",
                    value=default_config.get("cpu_offload", False),
                    info="Offload models to CPU to save VRAM (slower)"
                )
                
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎞️ Output")
                
                video_output = gr.Video(
                    label="Generated Video",
                    height=500,
                    autoplay=True
                )
                
                gr.Markdown("""
                ### 📝 Tips
                
                - **Prompt**: Be descriptive! Mention actions, movements, and details.
                - **Negative Prompt**: Specify what you want to avoid (e.g., "static, blurry, distorted").
                - **Inference Steps**: 4 steps is a good balance. More steps may improve quality.
                - **CFG Scale**: Higher values follow the prompt more closely, but can reduce quality.
                - **LoRA Strength**: Adjust if the style is too strong or too weak.
                - **Seed**: Use the same seed for reproducible results.
                
                ### 🎯 Example Prompts
                
                - "woman smiling and waving at camera"
                - "cat walking towards camera, sunny day"
                - "ocean waves gently moving, sunset"
                - "person turning head to look at camera"
                - "flower blooming in time-lapse"
                """)
        
        # Handle seed randomization
        def process_seed(seed):
            if seed == -1:
                import random
                return random.randint(0, 2**32 - 1)
            return int(seed)
        
        # Set up the generation pipeline
        generate_btn.click(
            fn=lambda *args: demo_instance.generate_video(
                *args[:3],  # image, prompt, negative_prompt
                process_seed(args[3]),  # seed
                *args[4:]  # rest of the settings
            ),
            inputs=[
                image_input,
                prompt_input,
                negative_prompt_input,
                seed_input,
                video_length_input,
                height_input,
                width_input,
                infer_steps_input,
                sample_shift_input,
                cfg_scale_high_input,
                cfg_scale_low_input,
                boundary_step_input,
                enable_loras_input,
                lora_strength_input,
                cpu_offload_input,
            ],
            outputs=video_output
        )
        
        # Example images
        gr.Markdown("### 🖼️ Example Images")
        gr.Examples(
            examples=[
                ["assets/inputs/imgs/img_0.jpg", "woman smiling and waving", ""],
                ["assets/inputs/imgs/pron_test.png", "woman looking at camera", ""],
            ],
            inputs=[image_input, prompt_input, negative_prompt_input],
            label="Click to load examples"
        )
    
    return demo


def main():
    """Main entry point for the Gradio demo."""
    parser = argparse.ArgumentParser(description="WAN2.2-MOE Quality Inference Gradio Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="prod_configs/wan_moe_i2v_a14b_quality.json",
        help="Path to quality config JSON"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/wan2.2_models/official_distill_repo",
        help="Path to model directory"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting WAN2.2-MOE Quality Inference Gradio Demo")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model path: {args.model_path}")
    
    demo = create_demo(args.config, args.model_path)
    
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )


if __name__ == "__main__":
    main()
