#!/usr/bin/env python3
"""Run WAN2.2-MOE i2v inference directly via Wan22MoeRunner.

This script mirrors ``run_wan22_i2v_simple.py`` but bypasses the generic
``lightx2v.infer`` entrypoint and instantiates the WAN runner directly. It keeps
all helpful defaults (metadata injection, negative prompt, etc.) so you can
reproduce the exact results of::

    python scripts/wan22/run_wan22_i2v_simple.py \
        --config_json configs/wan22/wan_moe_i2v_distill.json \
        --model_path ./models/wan2.2_models \
        --image_path /workspace/assets/inputs/imgs/img_0.jpg \
        --prompt "cat say meow" \
        --model_cls wan2.2_moe \
        --num_frames 41
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix
    resource = None

import torch
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all

WAN22_DEFAULT_MODEL_META = {
    "dim": 5120,
    "eps": 1e-6,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "in_dim": 36,
    "model_type": "i2v",
    "num_heads": 40,
    "num_layers": 40,
    "out_dim": 16,
    "text_len": 512,
}

DEFAULT_NEG_PROMPT = "Low quality, blurry, bad hands, distorted faces, static frame, watermark"
DEFAULT_OUTPUT_PATH = Path("./save_results/wan22_i2v_output.mp4")
DEFAULT_RAM_LIMIT_GB = 200


def enforce_memory_limit(limit_gb: int = DEFAULT_RAM_LIMIT_GB) -> None:
    """Clamp the process address-space limit to ``limit_gb`` gigabytes."""

    if resource is None:
        return

    limit_bytes = limit_gb * 1024 ** 3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = min(soft if soft != resource.RLIM_INFINITY else limit_bytes, limit_bytes)
    new_hard = min(hard if hard != resource.RLIM_INFINITY else limit_bytes, limit_bytes)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))


def load_and_patch_config(config_path: Path) -> Path:
    """Ensure the config contains WAN2.2 metadata before feeding it to Wan22MoeRunner.

    ``configs/wan22/wan_moe_i2v_distill.json`` only stores sampler/runtime knobs,
    so we inject the architecture metadata expected by the runner (dim, heads,
    etc.). The original ``run_wan22_i2v_simple.py`` also does this via ``setdefault``.
    When no patching is needed, we just return the original path.
    """

    config_path = config_path.resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict = json.load(f)

    changed = False
    for key, value in WAN22_DEFAULT_MODEL_META.items():
        if key not in config:
            config[key] = value
            changed = True

    if not changed:
        return config_path

    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with tmp:
        json.dump(config, tmp, indent=2)
    patched_path = Path(tmp.name)
    logger.debug(f"Patched config with WAN2.2 metadata: {patched_path}")
    return patched_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WAN2.2-MOE I2V direct runner invocation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, required=True, help="Path to WAN2.2 model directory")
    parser.add_argument("--config_json", type=Path, required=True, help="Base WAN22 config JSON")
    parser.add_argument("--image_path", type=Path, required=True, help="Input image for I2V")
    parser.add_argument("--prompt", type=str, required=True, help="Positive text prompt")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEG_PROMPT, help="Negative prompt")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the mp4 result")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_cls", type=str, default="wan2.2_moe", choices=["wan2.2_moe"], help="WAN model class")
    parser.add_argument("--task", type=str, default="i2v", choices=["i2v"], help="Task type")
    parser.add_argument("--num_frames", type=int, default=41, help="Legacy arg (config controls actual frames)")
    parser.add_argument("--return_result_tensor", action="store_true", help="Return tensor to caller instead of writing video")
    return parser


class DirectWan22MoeRunner(Wan22MoeRunner):
    """Thin helper around ``Wan22MoeRunner`` for scriptable direct inference.

    This subclass doesn't change model behavior—it simply wraps the common
    boilerplate (config printing, init, pipeline invocation) into a reusable
    object so other scripts can import and reuse it without going through the
    CLI runner.
    """

    def __init__(self, config, args_namespace):
        super().__init__(config)
        self._args = args_namespace

    def load_transformer(self):
        """Override to support LoRAs in any order using config 'name' field."""
        from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
        from lightx2v.models.networks.wan.model import WanModel
        from lightx2v.models.runners.wan.wan_runner import MultiModelStruct

        # encoder -> high_noise_model -> low_noise_model -> vae -> video_output
        high_noise_model = WanModel(
            self.high_noise_model_path,
            self.config,
            self.init_device,
            model_type="wan2.2_moe_high_noise",
        )
        low_noise_model = WanModel(
            self.low_noise_model_path,
            self.config,
            self.init_device,
            model_type="wan2.2_moe_low_noise",
        )

        if self.config.get("lora_configs") and self.config["lora_configs"]:
            assert not self.config.get("dit_quantized", False)

            for lora_config in self.config["lora_configs"]:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                model_name = lora_config.get("name", "")
                
                # Use config "name" field to determine target model, not filename
                # Create a new wrapper for each LoRA to support cumulative application
                if "high" in model_name.lower():
                    lora_wrapper = WanLoraWrapper(high_noise_model)
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
                elif "low" in model_name.lower():
                    lora_wrapper = WanLoraWrapper(low_noise_model)
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
                else:
                    raise ValueError(f"Unsupported LoRA config name: {model_name} (path: {lora_path})")

        return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary"])

    @classmethod
    def build(cls, args):
        """Seed RNGs, build config, and return an initialized runner instance."""

        logger.info("Seeding RNGs and building WAN2.2-MOE runner directly…")
        seed_all(args.seed)
        config = set_config(args)
        print_config(config)

        runner = cls(config, args)
        runner.init_modules()
        return runner

    def run(self):
        """Execute the regular WAN pipeline with the stored CLI args."""

        input_info = set_input_info(self._args)
        return self.run_pipeline(input_info)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    enforce_memory_limit()

    patched_config = load_and_patch_config(args.config_json)

    # ``set_config`` will read args.config_json, so point it to our patched copy.
    original_config_path = args.config_json
    args.config_json = patched_config

    # Downstream helpers expect plain strings (and json.dumps can't serialize Paths),
    # so convert every Path-bearing CLI argument to ``str``.
    path_fields = [
        "config_json",
        "model_path",
        "image_path",
        "output_path",
    ]
    for field in path_fields:
        value = getattr(args, field, None)
        if isinstance(value, Path):
            setattr(args, field, str(value))

    # ``set_input_info`` (and downstream runner logic) expect ``save_result_path``.
    # Mirror the semantics of the original simple script by forwarding the
    # ``--output_path`` argument to that attribute.
    if not hasattr(args, "save_result_path"):
        args.save_result_path = args.output_path

    # Maintain parity with infer.py signature (default False when flag unused).
    if not hasattr(args, "return_result_tensor"):
        args.return_result_tensor = False

    direct_runner = DirectWan22MoeRunner.build(args)
    direct_runner.run()

    logger.success(f"Video saved to {args.output_path}")

    # # Clean up temp file to avoid clutter.
    # if patched_config != original_config_path:
    #     patched_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
