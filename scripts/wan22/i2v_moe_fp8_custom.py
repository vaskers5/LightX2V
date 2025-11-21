from __future__ import annotations
from loguru import logger
try:
    from .i2v_moe_custom_base import Wan22MoeCustomRunner
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from i2v_moe_custom_base import Wan22MoeCustomRunner

class Wan22MoeFP8CustomRunner(Wan22MoeCustomRunner):
    """
    Custom runner for FP8 inference.
    Inherits from Wan22MoeCustomRunner which handles FP8 logic via config.
    This class can be used to enforce FP8 defaults or specific overrides.
    """
    def __init__(self, config):
        # Ensure FP8 settings are present if not set
        if not config.get("dit_quantized"):
            logger.warning("Wan22MoeFP8CustomRunner used but 'dit_quantized' is False in config. Enabling it.")
            config["dit_quantized"] = True
            # Default scheme if not set
            if not config.get("dit_quant_scheme"):
                config["dit_quant_scheme"] = "fp8-sgl" # Default to SGLang or whatever is best
        
        super().__init__(config)
        logger.info("Initialized Wan22MoeFP8CustomRunner")
