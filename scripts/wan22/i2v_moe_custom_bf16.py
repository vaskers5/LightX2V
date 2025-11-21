from __future__ import annotations
from loguru import logger
try:
    from .i2v_moe_custom_base import Wan22MoeCustomRunner
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from i2v_moe_custom_base import Wan22MoeCustomRunner

class Wan22MoeBF16CustomRunner(Wan22MoeCustomRunner):
    """
    Custom runner for BF16 inference.
    """
    def __init__(self, config):
        # Ensure FP8 is disabled
        if config.get("dit_quantized"):
            logger.warning("Wan22MoeBF16CustomRunner used but 'dit_quantized' is True. Disabling it.")
            config["dit_quantized"] = False
        
        config["use_bfloat16"] = True
        
        # Inject LightX LoRAs for BF16 inference
        # This forces the use of WanModel (default inference) + LoRA instead of WanDistillModel
        logger.info("Injecting LightX LoRAs for BF16 inference...")
        config["lora_configs"] = [
            {
                "name": "high_noise_model",
                "path": "models/wan2.2_models/loras/lightx_loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
                "strength": 1.0
            },
            {
                "name": "low_noise_model",
                "path": "models/wan2.2_models/loras/lightx_loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
                "strength": 1.0
            }
        ]
        
        super().__init__(config)
        logger.info("Initialized Wan22MoeBF16CustomRunner with LightX LoRAs")
