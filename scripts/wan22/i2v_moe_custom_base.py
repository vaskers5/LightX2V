#!/usr/bin/env python3
"""
Base module for WAN2.2-MOE i2v custom inference.
Contains shared classes and utilities for FP8 and BF16 inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from torch.distributed.tensor.device_mesh import init_device_mesh

from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct, Wan22MoeDistillRunner
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import ALL_INPUT_INFO_KEYS, set_input_info
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.utils import seed_all


class WeightAsyncStreamManager(object):
    def __init__(self, offload_granularity):
        self.offload_granularity = offload_granularity
        self.init_stream = torch.cuda.Stream(priority=0)
        self.cuda_load_stream = torch.cuda.Stream(priority=0)
        self.compute_stream = torch.cuda.Stream(priority=-1)

    def init_cuda_buffer(self, blocks_cuda_buffer=None, phases_cuda_buffer=None):
        if self.offload_granularity == "block":
            assert blocks_cuda_buffer is not None
            self.cuda_buffers = [blocks_cuda_buffer[i] for i in range(len(blocks_cuda_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cuda_buffer is not None
            self.cuda_buffers = [phases_cuda_buffer[i] for i in range(len(phases_cuda_buffer))]
        else:
            raise NotImplementedError

    def init_first_buffer(self, blocks, adapter_block_idx=None):
        if self.offload_granularity == "block":
            with torch.cuda.stream(self.init_stream):
                self.cuda_buffers[0].load_state_dict(blocks[0].state_dict(), 0, adapter_block_idx)
        else:
            with torch.cuda.stream(self.init_stream):
                self.cuda_buffers[0].load_state_dict(blocks[0].compute_phases[0].state_dict(), 0, adapter_block_idx)
        self.init_stream.synchronize()

    def prefetch_weights(self, block_idx, blocks, adapter_block_idx=None):
        with torch.cuda.stream(self.cuda_load_stream):
            self.cuda_buffers[1].load_state_dict(blocks[block_idx].state_dict(), block_idx, adapter_block_idx)

    def swap_blocks(self):
        self.cuda_load_stream.synchronize()
        self.compute_stream.synchronize()
        self.cuda_buffers[0], self.cuda_buffers[1] = (
            self.cuda_buffers[1],
            self.cuda_buffers[0],
        )

    def swap_weights(self):
        """Unified API for swapping buffers during weight offload."""
        if self.offload_granularity == "block":
            self.swap_blocks()
        elif self.offload_granularity == "phase":
            self.swap_phases()
        else:
            raise NotImplementedError(f"Unsupported granularity: {self.offload_granularity}")

    def prefetch_phase(self, block_idx, phase_idx, blocks, adapter_block_idx=None):
        with torch.cuda.stream(self.cuda_load_stream):
            self.cuda_buffers[phase_idx].load_state_dict(blocks[block_idx].compute_phases[phase_idx].state_dict(), block_idx, adapter_block_idx)

    def swap_phases(self):
        self.cuda_load_stream.synchronize()
        self.compute_stream.synchronize()


def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
        "teacache_thresh": 0.26,
        "use_ret_steps": False,
        "use_bfloat16": True,
        "lora_configs": None,  # List of dicts with 'path' and 'strength' keys
        "use_prompt_enhancer": False,
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "use_image_encoder": True,
    }
    default_config = LockableDict(default_config)
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items() if k not in ALL_INPUT_INFO_KEYS})
    
    if config.get("config_json", None) is not None:
        logger.info(f"Loading config from {config['config_json']}")
        with open(config["config_json"], "r") as f:
            loaded_config = json.load(f)
            
        # If the loaded config has a "config" key, merge its contents to the top level
        if "config" in loaded_config and isinstance(loaded_config["config"], dict):
            logger.info("Flattening 'config' section from JSON...")
            config.update(loaded_config["config"])
            # Also add other top-level keys
            for k, v in loaded_config.items():
                if k != "config":
                    config[k] = v
        else:
            config.update(loaded_config)
            
    return config


def set_parallel_config(config):
    if config["parallel"]:
        cfg_p_size = config["parallel"].get("cfg_p_size", 1)
        seq_p_size = config["parallel"].get("seq_p_size", 1)
        assert cfg_p_size * seq_p_size == dist.get_world_size(), f"cfg_p_size * seq_p_size must be equal to world_size"
        config["device_mesh"] = init_device_mesh("cuda", (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))

        if config["parallel"] and config["parallel"].get("seq_p_size", False) and config["parallel"]["seq_p_size"] > 1:
            config["seq_parallel"] = True

        if config.get("enable_cfg", False) and config["parallel"] and config["parallel"].get("cfg_p_size", False) and config["parallel"]["cfg_p_size"] > 1:
            config["cfg_parallel"] = True
        # warmup dist
        _a = torch.zeros([1]).to(f"cuda:{dist.get_rank()}")
        dist.all_reduce(_a)


def print_config(config):
    config_to_print = config.copy()
    config_to_print.pop("device_mesh", None)
    if config["parallel"]:
        if dist.get_rank() == 0:
            logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
    else:
        logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")


class NAGWanAttnProcessor:
    """
    Normalized Attention Guidance processor for WAN model attention layers.
    """
    
    def __init__(self, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        logger.debug(f"NAGWanAttnProcessor initialized (scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha})")
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Check if NAG should be applied
        apply_guidance = self.nag_scale > 1 and encoder_hidden_states is not None
        if apply_guidance:
            # NAG requires double batch: [positive_prompt, negative_prompt]
            if len(encoder_hidden_states) == 2 * len(hidden_states):
                batch_size = len(hidden_states)
            else:
                apply_guidance = False
        
        # Handle image encoder outputs (for I2V tasks)
        encoder_hidden_states_img = None
        if hasattr(attn, 'add_k_proj') and attn.add_k_proj is not None:
            # First 257 tokens are image features
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
            if apply_guidance:
                # Only use positive batch for image features
                encoder_hidden_states_img = encoder_hidden_states_img[:batch_size]
        
        # Default to self-attention if no cross-attention context
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Apply normalization if available
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Reshape to multi-head format [B, N, D] -> [B, H, N, D/H]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(
                    hidden_states.to(torch.float64).unflatten(3, (-1, 2))
                )
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        
        # Process image attention if present (I2V task)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            if hasattr(attn, 'norm_added_k'):
                key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            
            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            
            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        
        # Split K,V for NAG guidance if enabled
        if apply_guidance:
            key, key_negative = torch.chunk(key, 2, dim=0)
            value, value_negative = torch.chunk(value, 2, dim=0)
        
        # Compute positive attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        
        # Apply NAG if guidance is enabled
        if apply_guidance:
            # Compute negative attention
            hidden_states_negative = F.scaled_dot_product_attention(
                query, key_negative, value_negative, 
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_negative = hidden_states_negative.transpose(1, 2).flatten(2, 3)
            hidden_states_negative = hidden_states_negative.type_as(query)
            
            hidden_states_positive = hidden_states
            
            # Apply guidance: guided = positive * scale - negative * (scale - 1)
            hidden_states_guidance = (
                hidden_states_positive * self.nag_scale - 
                hidden_states_negative * (self.nag_scale - 1)
            )
            
            # Normalize guidance to prevent over-saturation
            norm_positive = torch.norm(
                hidden_states_positive, p=1, dim=-1, keepdim=True
            ).expand(*hidden_states_positive.shape)
            norm_guidance = torch.norm(
                hidden_states_guidance, p=1, dim=-1, keepdim=True
            ).expand(*hidden_states_guidance.shape)
            
            # Compute scaling ratio
            scale = norm_guidance / norm_positive
            scale = torch.nan_to_num(scale, 10)
            
            # Apply tau threshold: if scale > tau, normalize guidance
            mask = scale > self.nag_tau
            hidden_states_guidance[mask] = (
                hidden_states_guidance[mask] / (norm_guidance[mask] + 1e-7) * 
                norm_positive[mask] * self.nag_tau
            )
            
            # Blend normalized guidance with positive attention
            hidden_states = (
                hidden_states_guidance * self.nag_alpha + 
                hidden_states_positive * (1 - self.nag_alpha)
            )
        
        # Add image attention if present
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        
        # Project output
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class NAGModelWrapper:
    """
    Wrapper to inject NAG attention processors into WAN model.
    """
    
    def __init__(self, model: WanModel):
        self.model = model
        self.original_attn_processors = None
        self.nag_enabled = False
        logger.info("NAGModelWrapper initialized")
    
    def enable_nag(self, nag_scale: float = 1.5, nag_tau: float = 2.5, nag_alpha: float = 0.25):
        if self.nag_enabled:
            logger.warning("NAG already enabled, skipping")
            return
        
        logger.info(f"Enabling NAG (scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha})...")
        
        # Store NAG parameters
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        
        # Get transformer infer instance
        if not hasattr(self.model, 'transformer_infer'):
            logger.error("Model doesn't have transformer_infer attribute")
            return
        
        transformer_infer = self.model.transformer_infer
        
        # Store original cross_attn method
        self.original_infer_cross_attn = transformer_infer.infer_cross_attn
        
        # Replace with NAG-enhanced version
        transformer_infer.infer_cross_attn = self._create_nag_cross_attn(
            self.original_infer_cross_attn,
            transformer_infer,
            nag_scale,
            nag_tau,
            nag_alpha
        )
        
        self.nag_enabled = True
        logger.info(f"✓ NAG enabled successfully (cross_attn method patched)")
    
    def _create_nag_cross_attn(self, original_method, transformer_infer, nag_scale, nag_tau, nag_alpha):
        def nag_cross_attn(phase, x, context, y_out, gate_msa):
            # Check if NAG should be applied
            apply_guidance = nag_scale > 1 and hasattr(self, 'negative_context')
            
            if not apply_guidance:
                # No NAG, use original method
                return original_method(phase, x, context, y_out, gate_msa)
            
            # Get positive and negative contexts
            context_positive = context
            context_negative = self.negative_context.squeeze(0) if self.negative_context.dim() == 3 else self.negative_context
            
            # Pad negative context if feature dimension mismatches (e.g. 5120 vs 4096)
            if context_positive.shape[-1] != context_negative.shape[-1]:
                diff = context_positive.shape[-1] - context_negative.shape[-1]
                if diff > 0:
                    padding = torch.zeros(
                        *context_negative.shape[:-1], diff,
                        dtype=context_negative.dtype,
                        device=context_negative.device
                    )
                    context_negative = torch.cat([context_negative, padding], dim=-1)
            
            # Handle image context if present
            should_slice = (
                transformer_infer.task in ["i2v", "flf2v", "animate", "s2v"] and 
                transformer_infer.config.get("use_image_encoder", True) and
                context_positive.shape[0] > 257
            )
            
            if should_slice:
                # Slice along first dimension (sequence)
                context_img = context_positive[:257]  # [257, D]
                context_positive = context_positive[257:]  # [seq-257, D]
                context_negative = context_negative[257:]  # [seq-257, D]
            else:
                context_img = None
            
            # Compute positive attention
            attn_out_positive = self._compute_cross_attn(
                transformer_infer, phase, x, context_positive, context_img, y_out, gate_msa
            )
            
            # Compute negative attention
            attn_out_negative = self._compute_cross_attn(
                transformer_infer, phase, x, context_negative, None, y_out, gate_msa
            )
            
            # Apply NAG guidance
            attn_guided = (
                attn_out_positive * nag_scale -
                attn_out_negative * (nag_scale - 1)
            )
            
            # Normalize guidance to prevent over-saturation
            norm_positive = torch.norm(
                attn_out_positive, p=1, dim=-1, keepdim=True
            ).expand(*attn_out_positive.shape)
            norm_guidance = torch.norm(
                attn_guided, p=1, dim=-1, keepdim=True
            ).expand(*attn_guided.shape)
            
            # Compute scaling ratio
            scale = norm_guidance / (norm_positive + 1e-7)
            scale = torch.nan_to_num(scale, 10)
            
            # Apply tau threshold: if scale > tau, normalize guidance
            mask = scale > nag_tau
            attn_guided[mask] = (
                attn_guided[mask] / (norm_guidance[mask] + 1e-7) *
                norm_positive[mask] * nag_tau
            )
            
            # Blend normalized guidance with positive attention
            attn_out = (
                attn_guided * nag_alpha +
                attn_out_positive * (1 - nag_alpha)
            )
            
            # Calculate updated x to return (same logic as in _compute_cross_attn)
            if transformer_infer.sensitive_layer_dtype != transformer_infer.infer_dtype:
                x_updated = x.to(transformer_infer.sensitive_layer_dtype) + y_out.to(transformer_infer.sensitive_layer_dtype) * gate_msa.squeeze()
            else:
                x_updated = x + y_out * gate_msa.squeeze()
            
            return x_updated, attn_out
        
        return nag_cross_attn
    
    def _compute_cross_attn(self, transformer_infer, phase, x, context, context_img, y_out, gate_msa):
        if transformer_infer.sensitive_layer_dtype != transformer_infer.infer_dtype:
            x_norm = x.to(transformer_infer.sensitive_layer_dtype) + y_out.to(transformer_infer.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x_norm = x + y_out * gate_msa.squeeze()
        
        norm3_out = phase.norm3.apply(x_norm)
        
        if transformer_infer.sensitive_layer_dtype != transformer_infer.infer_dtype:
            context = context.to(transformer_infer.infer_dtype)
            if context_img is not None:
                context_img = context_img.to(transformer_infer.infer_dtype)
        
        n, d = transformer_infer.num_heads, transformer_infer.head_dim
        
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
        v = phase.cross_attn_v.apply(context).view(-1, n, d)
        
        cu_seqlens_q, cu_seqlens_k = transformer_infer._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device),
        )
        
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
            model_cls=transformer_infer.config["model_cls"],
        )
        
        # Add image attention if present
        if context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
            
            cu_seqlens_q, cu_seqlens_k = transformer_infer._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )
            
            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
                model_cls=transformer_infer.config["model_cls"],
            )
            attn_out = attn_out + img_attn_out
        
        y = phase.cross_attn_o.apply(attn_out)
        return y
    
    def disable_nag(self):
        if not self.nag_enabled:
            logger.warning("NAG not enabled, skipping")
            return
        
        logger.info("Disabling NAG...")
        
        # Restore original method
        if hasattr(self.model, 'transformer_infer') and hasattr(self, 'original_infer_cross_attn'):
            self.model.transformer_infer.infer_cross_attn = self.original_infer_cross_attn
            delattr(self, 'original_infer_cross_attn')
        
        self.nag_enabled = False
        logger.info("✓ NAG disabled")
    
    def __getattr__(self, name):
        return getattr(self.model, name)


class CustomWeightAsyncStreamManager(WeightAsyncStreamManager):
    """
    Custom weight manager that extends WeightAsyncStreamManager with swap_weights method.
    """
    
    def __init__(self, offload_granularity):
        super().__init__(offload_granularity)
    
    def swap_weights(self):
        if self.offload_granularity == "block":
            self.swap_blocks()
        elif self.offload_granularity == "phase":
            self.swap_phases()
        else:
            raise ValueError(f"Unknown offload granularity: {self.offload_granularity}")


class MotionAmplitudeProcessor:
    """
    Enhanced latent processor to fix slow-motion issues in 4-step LoRAs.
    """
    
    def __init__(self, motion_amplitude=1.15, enable=True):
        self.motion_amplitude = motion_amplitude
        self.enable = enable
        logger.info(f"MotionAmplitudeProcessor initialized (amplitude={motion_amplitude}, enabled={enable})")
    
    @torch.no_grad()
    def apply_motion_enhancement(self, vae_latent):
        if not self.enable or self.motion_amplitude <= 1.0:
            return vae_latent
        
        logger.debug(f"Applying motion enhancement (amplitude={self.motion_amplitude})...")
        
        # Extract first temporal frame (real) and subsequent frames (gray/zero placeholders)
        base_latent = vae_latent[:, 0:1, :, :]      # [C, 1, H, W] - first frame
        gray_latent = vae_latent[:, 1:, :, :]       # [C, T-1, H, W] - gray frames
        
        # Calculate motion difference
        diff = gray_latent - base_latent
        
        # Preserve brightness by extracting and re-adding mean
        diff_mean = diff.mean(dim=(2, 3), keepdim=True)
        diff_centered = diff - diff_mean
        
        # Amplify centered motion and restore mean
        scaled_latent = base_latent + diff_centered * self.motion_amplitude + diff_mean
        
        # Clamp to prevent artifacts (empirically determined range for latent space)
        scaled_latent = torch.clamp(scaled_latent, -6, 6)
        
        # Combine first frame with enhanced frames along temporal dimension (dim=1)
        enhanced_latent = torch.cat([base_latent, scaled_latent], dim=1)
        
        return enhanced_latent


class EnhancedVAEEncoder:
    """
    Wrapper for VAE encoder that applies motion amplitude enhancement.
    """
    
    def __init__(self, vae_encoder, motion_processor=None):
        self.vae_encoder = vae_encoder
        self.motion_processor = motion_processor or MotionAmplitudeProcessor(enable=False)
    
    def encode(self, vae_input):
        vae_latent = self.vae_encoder.encode(vae_input)
        
        if self.motion_processor.enable:
            vae_latent = self.motion_processor.apply_motion_enhancement(vae_latent)
        
        return vae_latent
    
    def __getattr__(self, name):
        return getattr(self.vae_encoder, name)


class FrameInterpolator:
    """
    Frame interpolation processor using RIFE model for smooth video upsampling.
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        
        logger.info(f"FrameInterpolator initialized (device={self.device})")
    
    def _load_model(self):
        if self.model is not None:
            return
        
        logger.info(f"Loading RIFE model from {self.model_path}...")
        
        from lightx2v.models.vfi.rife.rife_comfyui_wrapper import RIFEWrapper
        self.model = RIFEWrapper(self.model_path, device=self.device)
        logger.info("✓ RIFE model loaded successfully")

    @torch.no_grad()
    def interpolate(
        self,
        video_tensor: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if source_fps >= target_fps:
            logger.warning(f"Target FPS ({target_fps}) <= source FPS ({source_fps}), skipping interpolation")
            return video_tensor
        
        self._load_model()
        
        logger.info(f"Interpolating video: {source_fps} FPS → {target_fps} FPS (scale={scale})")
        
        is_pytorch_format = video_tensor.shape[1] == 3 or video_tensor.shape[1] == 1
        
        if is_pytorch_format:
            video_tensor = video_tensor.permute(0, 2, 3, 1)
        
        original_dtype = video_tensor.dtype
        if video_tensor.dtype != torch.float32:
            video_tensor = video_tensor.to(torch.float32)
        
        interpolated = self.model.interpolate_frames(
            video_tensor,
            source_fps=source_fps,
            target_fps=target_fps,
            scale=scale
        )
        
        if interpolated.dtype != original_dtype:
            interpolated = interpolated.to(original_dtype)
        
        if is_pytorch_format:
            interpolated = interpolated.permute(0, 3, 1, 2)
        
        logger.info(f"✓ Interpolation complete: {video_tensor.shape[0]} → {interpolated.shape[0]} frames")
        
        return interpolated
    
    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ FrameInterpolator resources released")


class VideoPackager:
    """
    Video packaging utility for encoding tensors to video files.
    """
    
    def __init__(
        self,
        output_path: str,
        fps: float = 24.0,
        codec: str = "mp4v",
        quality: Optional[int] = None,
        use_ffmpeg: bool = False,
    ):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.use_ffmpeg = use_ffmpeg
        self.writer = None
        self.frame_count = 0
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoPackager initialized: {output_path} ({fps} FPS, codec={codec})")
    
    def _init_opencv_writer(self, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {self.output_path}")
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        if tensor.shape[0] == 3:  # [C, H, W] format
            tensor = tensor.permute(1, 2, 0)
        
        frame = tensor.numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def write_frame(self, frame_tensor: torch.Tensor):
        if self.writer is None:
            if frame_tensor.shape[0] == 3:  # [C, H, W]
                height, width = frame_tensor.shape[1:3]
            else:  # [H, W, C]
                height, width = frame_tensor.shape[0:2]
            
            self._init_opencv_writer(width, height)
        
        frame = self._tensor_to_frame(frame_tensor)
        self.writer.write(frame)
        self.frame_count += 1
    
    def write_video(self, video_tensor: torch.Tensor, show_progress: bool = True):
        num_frames = video_tensor.shape[0]
        
        logger.info(f"Writing {num_frames} frames to {self.output_path}...")
        
        for i in range(num_frames):
            self.write_frame(video_tensor[i])
            
            if show_progress and (i + 1) % 30 == 0:
                logger.debug(f"  Progress: {i + 1}/{num_frames} frames written")
        
        logger.info(f"✓ Video writing complete: {self.frame_count} frames")
    
    def finalize(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"✓ Video finalized: {self.output_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
    
    @staticmethod
    def quick_save(
        video_tensor: torch.Tensor,
        output_path: str,
        fps: float = 24.0,
        codec: str = "mp4v",
    ):
        with VideoPackager(output_path, fps=fps, codec=codec) as packager:
            packager.write_video(video_tensor)


class WanLoraCustomWrapper:
    def __init__(self, wan_model):
        self.model = wan_model
        self.lora_metadata = {}
        self.override_dict = {}  # On CPU

    def load_lora(self, lora_path, lora_name=None):
        if lora_name is None:
            lora_name = os.path.basename(lora_path).split(".")[0]

        if lora_name in self.lora_metadata:
            logger.info(f"LoRA {lora_name} already loaded, skipping...")
            return lora_name

        self.lora_metadata[lora_name] = {"path": lora_path}
        logger.info(f"Registered LoRA metadata for: {lora_name} from {lora_path}")

        return lora_name

    def _load_lora_file(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(GET_DTYPE()) for key in f.keys()}
        return tensor_dict

    def apply_lora(self, lora_name, alpha=1.0):
        if lora_name not in self.lora_metadata:
            logger.info(f"LoRA {lora_name} not found. Please load it first.")

        if not hasattr(self.model, "original_weight_dict"):
            logger.error("Model does not have 'original_weight_dict'. Cannot apply LoRA.")
            return False

        lora_weights = self._load_lora_file(self.lora_metadata[lora_name]["path"])
        weight_dict = self.model.original_weight_dict
        self._apply_lora_weights(weight_dict, lora_weights, alpha)
        self.model._apply_weights(weight_dict)

        logger.info(f"Applied LoRA: {lora_name} with alpha={alpha}")
        del lora_weights
        return True

    @torch.no_grad()
    def _apply_lora_weights(self, weight_dict, lora_weights, alpha):
        import re
        
        def normalize_key(key):
            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model."):]
            elif key.startswith("lora_unet_"):
                key = key.replace("lora_unet_", "blocks.")
            
            key = re.sub(r"\.(?:lora_(?:down|up|A|B)|alpha)(?:\.weight)?$", "", key)
            
            if key.endswith(".weight"):
                key = key[:-7]
                
            return key

        model_canonical_map = {}
        for k in weight_dict.keys():
            canon = normalize_key(k)
            model_canonical_map[canon] = k

        lora_groups = {}
        for key, tensor in lora_weights.items():
            canon = normalize_key(key)
            if canon not in lora_groups:
                lora_groups[canon] = {}
            
            if "lora_down" in key or "lora_A" in key:
                lora_groups[canon]["down"] = tensor
            elif "lora_up" in key or "lora_B" in key:
                lora_groups[canon]["up"] = tensor
            elif "alpha" in key:
                lora_groups[canon]["alpha"] = tensor

        applied_count = 0
        
        for canon, groups in lora_groups.items():
            target_key = model_canonical_map.get(canon)
            if not target_key:
                continue
            
            if "down" in groups and "up" in groups:
                if target_key not in self.override_dict:
                    self.override_dict[target_key] = weight_dict[target_key].clone().cpu()
                
                param = weight_dict[target_key]
                
                is_fp8 = param.dtype == torch.float8_e4m3fn
                original_dtype = param.dtype
                if is_fp8:
                    param.data = param.data.to(torch.bfloat16)
                
                down = groups["down"].to(param.device, param.dtype)
                up = groups["up"].to(param.device, param.dtype)
                
                current_alpha = alpha
                if "alpha" in groups:
                    lora_alpha_tensor = groups["alpha"]
                    lora_alpha_val = lora_alpha_tensor.item() if lora_alpha_tensor.numel() == 1 else lora_alpha_tensor[0].item()
                    rank = down.shape[0] 
                    current_alpha *= (lora_alpha_val / rank)
                
                if param.shape == (up.shape[0], down.shape[1]):
                     update = torch.matmul(up, down) * current_alpha
                     param.data += update
                     applied_count += 1
                elif param.dim() == 4 and down.dim() == 4 and up.dim() == 4:
                     if down.shape[-2:] == (1, 1) and up.shape[-2:] == (1, 1):
                         d_s = down.squeeze(-1).squeeze(-1)
                         u_s = up.squeeze(-1).squeeze(-1)
                         update = torch.matmul(u_s, d_s) * current_alpha
                         update = update.unsqueeze(-1).unsqueeze(-1)
                         if param.shape == update.shape:
                             param.data += update
                             applied_count += 1
                
                if is_fp8:
                    param.data = param.data.to(original_dtype)

        logger.info(f"Applied {applied_count} LoRA weight adjustments")
        if applied_count == 0:
            logger.info(
                "Warning: No LoRA weights were applied. Please verify the LoRA weight file and naming conventions."
            )

    @torch.no_grad()
    def remove_lora(self):
        logger.info(f"Removing LoRA ...")

        restored_count = 0
        for k, v in self.override_dict.items():
            self.model.original_weight_dict[k] = v.to(self.model.device)
            restored_count += 1

        logger.info(f"LoRA removed, restored {restored_count} weights")

        self.model._apply_weights(self.model.original_weight_dict)

        torch.cuda.empty_cache()
        gc.collect()

        self.lora_metadata = {}
        self.override_dict = {}

    def list_loaded_loras(self):
        return list(self.lora_metadata.keys())


class CustomMultiDistillModelStruct(MultiDistillModelStruct):
    """
    Custom MultiDistillModelStruct that enforces model offloading for VRAM optimization.
    """
    def __init__(self, model_list, config, boundary_step_index=2):
        super().__init__(model_list, config, boundary_step_index)
        self.cur_model_index = -1

    def get_current_model_index(self):
        # Determine which model to use based on step index
        if self.scheduler.step_index < self.boundary_step_index:
            # High noise phase
            target_index = 0
            logger.info(f"using - HIGH - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][0]
        else:
            # Low noise phase
            target_index = 1
            logger.info(f"using - LOW - noise model at step_index {self.scheduler.step_index + 1}")
            self.scheduler.sample_guide_scale = self.config["sample_guide_scale"][1]

        # Perform switching if needed
        if self.cur_model_index != target_index:
            logger.info(f"Switching model: {self.cur_model_index} -> {target_index}")
            
            # Only perform offloading/loading if cpu_offload is enabled
            if self.config.get("cpu_offload", False):
                # Offload the previous model if it was loaded
                if self.cur_model_index != -1:
                    logger.info(f"Offloading model {self.cur_model_index} to CPU")
                    self.offload_cpu(self.cur_model_index)
                    torch.cuda.empty_cache()
                
                # Load the target model
                logger.info(f"Loading model {target_index} to CUDA")
                self.to_cuda(target_index)
            
            self.cur_model_index = target_index


class Wan22MoeCustomRunner(Wan22MoeDistillRunner):
    """Custom WAN2.2-MOE Distill runner with enhanced LoRA handling and motion optimization."""

    def __init__(self, config):
        if "denoising_steps" in config:
            steps_config = config["denoising_steps"]
            
            if "steps" in steps_config and "boundary" in steps_config:
                steps = steps_config["steps"]
                boundary = steps_config["boundary"]
                boundary_step_index = int(steps * boundary)
                num_train_timesteps = 1000
                step_size = num_train_timesteps // steps
                all_steps = [num_train_timesteps - i * step_size for i in range(steps)]
                high_steps = all_steps[:boundary_step_index]
                low_steps = all_steps[boundary_step_index:]
                
                config["denoising_step_list_high"] = high_steps
                config["denoising_step_list_low"] = low_steps
                config["denoising_step_list"] = all_steps
                config["boundary_step_index"] = boundary_step_index
                config["infer_steps"] = steps
                config["boundary"] = boundary
                
                logger.info(f"Auto-calculated denoising steps: {steps} steps, boundary {boundary}")
            else:
                if "high" in steps_config:
                    config["denoising_step_list_high"] = steps_config["high"]
                if "low" in steps_config:
                    config["denoising_step_list_low"] = steps_config["low"]
                if "list" in steps_config:
                    config["denoising_step_list"] = steps_config["list"]
                if "boundary_step_index" in steps_config:
                    config["boundary_step_index"] = steps_config["boundary_step_index"]
                
        super().__init__(config)
        
        if "motion_enhancement" in self.config:
            motion_config = self.config["motion_enhancement"]
            motion_amplitude = motion_config.get("amplitude", 1.15)
            enable_motion = motion_config.get("enabled", True)
        else:
            motion_amplitude = self.config.get("motion_amplitude", 1.15)
            enable_motion = self.config.get("enable_motion_enhancement", True)

        self.motion_processor = MotionAmplitudeProcessor(
            motion_amplitude=motion_amplitude,
            enable=enable_motion
        )
        
        self.nag_wrapper = None
        self.nag_config = self.config.get("nag", {})
        if self.nag_config.get("enabled", False):
            nag_scale = self.nag_config.get("scale", 1.5)
            nag_tau = self.nag_config.get("tau", 2.5)
            nag_alpha = self.nag_config.get("alpha", 0.25)
            logger.info(f"NAG enabled: scale={nag_scale}, tau={nag_tau}, alpha={nag_alpha}")
        else:
            logger.info("NAG disabled (set nag.enabled=true in config to enable)")
        
        logger.info(f"Initialized {self.__class__.__name__} (Distilled version)")
    
    def _patch_t5_offload_manager(self):
        if not hasattr(self, 'text_encoders') or not self.text_encoders:
            return
        
        t5_encoder = self.text_encoders[0]
        
        if hasattr(t5_encoder, 'model') and hasattr(t5_encoder.model, 'offload_manager'):
            original_manager = t5_encoder.model.offload_manager
            custom_manager = CustomWeightAsyncStreamManager(
                offload_granularity=original_manager.offload_granularity
            )
            for attr_name in dir(original_manager):
                if not attr_name.startswith('_') and attr_name not in ['swap_weights', 'swap_blocks', 'swap_phases']:
                    try:
                        attr_value = getattr(original_manager, attr_name)
                        if not callable(attr_value):
                            setattr(custom_manager, attr_name, attr_value)
                    except AttributeError:
                        pass
            t5_encoder.model.offload_manager = custom_manager
            logger.info("✓ Patched T5 encoder with CustomWeightAsyncStreamManager")
        elif hasattr(t5_encoder, 'text_encoder') and hasattr(t5_encoder.text_encoder, 'offload_manager'):
            original_manager = t5_encoder.text_encoder.offload_manager
            custom_manager = CustomWeightAsyncStreamManager(
                offload_granularity=original_manager.offload_granularity
            )
            for attr_name in dir(original_manager):
                if not attr_name.startswith('_') and attr_name not in ['swap_weights', 'swap_blocks', 'swap_phases']:
                    try:
                        attr_value = getattr(original_manager, attr_name)
                        if not callable(attr_value):
                            setattr(custom_manager, attr_name, attr_value)
                    except AttributeError:
                        pass
            t5_encoder.text_encoder.offload_manager = custom_manager
            logger.info("✓ Patched T5 encoder with CustomWeightAsyncStreamManager")
    
    def init_modules(self):
        super().init_modules()
        
        if self.config.get("t5_cpu_offload", False):
            logger.info("T5 CPU offload is enabled, patching offload manager...")
            self._patch_t5_offload_manager()
        
        if self.nag_config.get("enabled", False):
            logger.info("Initializing NAG wrapper for model...")
            if isinstance(self.model, MultiDistillModelStruct):
                self.nag_wrapper = NAGModelWrapper(self.model.model[0])
                logger.info("NAG wrapper applied to high noise model")
            else:
                self.nag_wrapper = NAGModelWrapper(self.model)
                logger.info("NAG wrapper applied to model")
        
        logger.info("✓ Custom runner modules initialized successfully!")

    def load_transformer(self):
        is_quantized = self.config.get("dit_quantized", False)
        quant_scheme = self.config.get("dit_quant_scheme", "")
        
        if is_quantized:
            logger.info(f"Loading models with FP8 quantization ({quant_scheme})...")
        else:
            logger.info("Loading distilled transformer models (high noise + low noise)...")
        
        use_high_lora, use_low_lora = False, False
        if self.config.get("lora_configs") and self.config["lora_configs"]:
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "high" in lora_name.lower():
                    use_high_lora = True
                elif "low" in lora_name.lower():
                    use_low_lora = True
        
        if is_quantized and (use_high_lora or use_low_lora):
            logger.info("✓ Using FP8 quantization + LoRAs (LoRAs applied on top of quantized models)")

        if use_high_lora:
            if is_quantized:
                logger.info(f"  Loading HIGH noise FP8 model ({quant_scheme}) with LoRA support...")
                high_noise_model = WanDistillModel(
                    self.high_noise_model_path,
                    self.config,
                    self.init_device,
                    model_type="wan2.2_moe_high_noise",
                )
            else:
                logger.info("  Loading HIGH noise model with LoRA support (no quantization)...")
                # Create a temporary config copy to avoid modifying the global config
                temp_config = self.config.copy()
                # Ensure dit_quantized is False for WanModel initialization in BF16 mode
                if temp_config.get("dit_quantized", False):
                    logger.warning("  Forcing dit_quantized=False for WanModel initialization (BF16 mode)")
                    temp_config["dit_quantized"] = False
                # Ensure dit_quant_scheme is Default to avoid assertion error in WanTransformerWeights
                if temp_config.get("dit_quant_scheme", "Default") != "Default":
                    logger.warning("  Forcing dit_quant_scheme='Default' for WanModel initialization (BF16 mode)")
                    temp_config["dit_quant_scheme"] = "Default"
                
                high_noise_model = WanModel(
                    self.high_noise_model_path,
                    temp_config,
                    self.init_device,
                    model_type="wan2.2_moe_high_noise",
                )
            
            high_lora_wrapper = WanLoraCustomWrapper(high_noise_model)
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "high" in lora_name.lower():
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_id = high_lora_wrapper.load_lora(lora_path)
                    high_lora_wrapper.apply_lora(lora_id, strength)
                    logger.info(f"    ✓ Applied LoRA '{lora_name}' (strength={strength})")
        else:
            if is_quantized:
                logger.info(f"  Loading HIGH noise FP8 model ({quant_scheme})...")
            else:
                logger.info("  Loading HIGH noise DISTILLED model (no LoRA)...")
            high_noise_model = WanDistillModel(
                self.high_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_high_noise",
            )
        logger.info(f"  ✓ High noise model loaded from {self.high_noise_model_path}")

        if use_low_lora:
            if is_quantized:
                logger.info(f"  Loading LOW noise FP8 model ({quant_scheme}) with LoRA support...")
                low_noise_model = WanDistillModel(
                    self.low_noise_model_path,
                    self.config,
                    self.init_device,
                    model_type="wan2.2_moe_low_noise",
                )
            else:
                logger.info("  Loading LOW noise model with LoRA support (no quantization)...")
                # Create a temporary config copy to avoid modifying the global config
                temp_config = self.config.copy()
                # Ensure dit_quantized is False for WanModel initialization in BF16 mode
                if temp_config.get("dit_quantized", False):
                    logger.warning("  Forcing dit_quantized=False for WanModel initialization (BF16 mode)")
                    temp_config["dit_quantized"] = False
                # Ensure dit_quant_scheme is Default to avoid assertion error in WanTransformerWeights
                if temp_config.get("dit_quant_scheme", "Default") != "Default":
                    logger.warning("  Forcing dit_quant_scheme='Default' for WanModel initialization (BF16 mode)")
                    temp_config["dit_quant_scheme"] = "Default"
                    
                low_noise_model = WanModel(
                    self.low_noise_model_path,
                    temp_config,
                    self.init_device,
                    model_type="wan2.2_moe_low_noise",
                )
            
            low_lora_wrapper = WanLoraCustomWrapper(low_noise_model)
            for lora_config in self.config["lora_configs"]:
                lora_name = lora_config.get("name", "")
                if "low" in lora_name.lower():
                    lora_path = lora_config["path"]
                    strength = lora_config.get("strength", 1.0)
                    lora_id = low_lora_wrapper.load_lora(lora_path)
                    low_lora_wrapper.apply_lora(lora_id, strength)
                    logger.info(f"    ✓ Applied LoRA '{lora_name}' (strength={strength})")
        else:
            if is_quantized:
                logger.info(f"  Loading LOW noise FP8 model ({quant_scheme})...")
            else:
                logger.info("  Loading LOW noise DISTILLED model (no LoRA)...")
            low_noise_model = WanDistillModel(
                self.low_noise_model_path,
                self.config,
                self.init_device,
                model_type="wan2.2_moe_low_noise",
            )
        logger.info(f"  ✓ Low noise model loaded from {self.low_noise_model_path}")

        # Wrap models in CustomMultiDistillModelStruct for two-stage distilled inference with offloading
        multi_model = CustomMultiDistillModelStruct(
            [high_noise_model, low_noise_model],
            self.config,
            self.config["boundary_step_index"]
        )
        
        return multi_model

    def run_text_encoder(self, input_info):
        t5_offload = self.config.get("t5_cpu_offload", False)
        nag_enabled = self.nag_config.get("enabled", False)
        
        if t5_offload:
            logger.info("Loading T5 encoder to GPU for text encoding...")
            if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
                self.text_encoders = self.load_text_encoder()
            if hasattr(self.text_encoders[0], 'model'):
                self.text_encoders[0].model = self.text_encoders[0].model.cuda()
            elif hasattr(self.text_encoders[0], 'text_encoder'):
                self.text_encoders[0].text_encoder = self.text_encoders[0].text_encoder.cuda()
        
        result = super().run_text_encoder(input_info)
        
        if nag_enabled and self.nag_wrapper is not None:
            logger.info("Encoding NAG negative prompt...")
            
            nag_negative_prompt = self.nag_config.get("negative_prompt", "")
            if not nag_negative_prompt and hasattr(input_info, 'negative_prompt'):
                nag_negative_prompt = input_info.negative_prompt
            
            if not nag_negative_prompt:
                logger.warning("NAG enabled but no negative prompt provided, using empty string")
                nag_negative_prompt = ""
            
            from copy import deepcopy
            negative_input_info = deepcopy(input_info)
            negative_input_info.prompt = nag_negative_prompt
            
            negative_result = super().run_text_encoder(negative_input_info)
            
            if 'context' in negative_result:
                negative_ctx = negative_result['context']
                
                if self.config["task"] in ["i2v", "flf2v", "animate", "s2v"] and self.config.get("use_image_encoder", True):
                    batch_size, text_len, hidden_dim = negative_ctx.shape
                    image_len = 257
                    
                    zero_padding = torch.zeros(
                        batch_size, image_len, hidden_dim,
                        dtype=negative_ctx.dtype,
                        device=negative_ctx.device
                    )
                    
                    negative_ctx = torch.cat([zero_padding, negative_ctx], dim=1)
                    logger.info(f"NAG negative context padded with {image_len} zero image features")
                
                self.nag_wrapper.negative_context = negative_ctx
                logger.info(f"NAG negative context stored: {self.nag_wrapper.negative_context.shape}")
            
            del negative_result
        
        if t5_offload:
            logger.info("Offloading T5 encoder to CPU...")
            if hasattr(self.text_encoders[0], 'model'):
                self.text_encoders[0].model = self.text_encoders[0].model.cpu()
            elif hasattr(self.text_encoders[0], 'text_encoder'):
                self.text_encoders[0].text_encoder = self.text_encoders[0].text_encoder.cpu()
            
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ T5 offloaded, GPU memory freed")
        
        return result

    def get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None):
        h = lat_h * self.config["vae_stride"][1]
        w = lat_w * self.config["vae_stride"][2]
        msk = torch.ones(
            1,
            self.config["target_video_length"],
            lat_h,
            lat_w,
            device=torch.device("cuda"),
        )
        if last_frame is not None:
            msk[:, 1:-1] = 0
        else:
            msk[:, 1:] = 0

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        vae_offload = self.config.get("vae_cpu_offload", False)
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()
        
        if vae_offload:
            logger.debug("Loading VAE encoder to GPU...")
            if hasattr(self.vae_encoder, 'encoder'):
                self.vae_encoder.encoder = self.vae_encoder.encoder.cuda()
            elif hasattr(self.vae_encoder, 'vae'):
                self.vae_encoder.vae = self.vae_encoder.vae.cuda()

        if last_frame is not None:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 2, h, w),
                    torch.nn.functional.interpolate(last_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                ],
                dim=1,
            ).cuda()
        else:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 1, h, w),
                ],
                dim=1,
            ).cuda()

        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))
        
        if self.motion_processor.enable and self.motion_processor.motion_amplitude > 1.0:
            logger.debug("Applying motion amplitude enhancement to VAE latents...")
            vae_encoder_out = self.motion_processor.apply_motion_enhancement(vae_encoder_out)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        elif vae_offload:
            logger.debug("Offloading VAE encoder to CPU...")
            if hasattr(self.vae_encoder, 'encoder'):
                self.vae_encoder.encoder = self.vae_encoder.encoder.cpu()
            elif hasattr(self.vae_encoder, 'vae'):
                self.vae_encoder.vae = self.vae_encoder.vae.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("✓ VAE encoder offloaded")
            
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
        return vae_encoder_out

    def run_vae_decoder(self, latents):
        vae_offload = self.config.get("vae_cpu_offload", False)
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        
        if vae_offload:
            logger.info("Loading VAE decoder to GPU for latent decoding...")
            if hasattr(self.vae_decoder, 'decoder'):
                self.vae_decoder.decoder = self.vae_decoder.decoder.cuda()
            elif hasattr(self.vae_decoder, 'vae'):
                self.vae_decoder.vae = self.vae_decoder.vae.cuda()
        
        images = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        elif vae_offload:
            logger.info("Offloading VAE decoder to CPU...")
            if hasattr(self.vae_decoder, 'decoder'):
                self.vae_decoder.decoder = self.vae_decoder.decoder.cpu()
            elif hasattr(self.vae_decoder, 'vae'):
                self.vae_decoder.vae = self.vae_decoder.vae.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ VAE decoder offloaded, GPU memory freed")
        
        return images
    
    def cleanup_after_inference(self):
        import gc
        
        logger.info("Cleaning up GPU memory...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        logger.info("✓ Memory cleanup complete")

    def run_pipeline(self, input_info):
        logger.info("=" * 80)
        logger.info("Starting custom DISTILLED inference pipeline...")
        logger.info(f"Prompt: {input_info.prompt}")
        logger.info(f"Image: {input_info.image_path}")
        if self.nag_config.get("enabled", False):
            logger.info(f"NAG: ENABLED (scale={self.nag_config.get('scale', 1.5)})")
        logger.info("=" * 80)
        
        if self.nag_wrapper is not None and not self.nag_wrapper.nag_enabled:
            self.nag_wrapper.enable_nag(
                nag_scale=self.nag_config.get("scale", 1.5),
                nag_tau=self.nag_config.get("tau", 2.5),
                nag_alpha=self.nag_config.get("alpha", 0.25)
            )
        
        frame_interp_config = self.config.get("frame_interpolation", {})
        should_interpolate = frame_interp_config.get("enabled", False)
        
        original_return_tensor = input_info.return_result_tensor
        original_save_path = input_info.save_result_path
        if should_interpolate and not input_info.return_result_tensor:
            input_info.return_result_tensor = True
            temp_save_path = input_info.save_result_path
            input_info.save_result_path = None
        
        result = super().run_pipeline(input_info)
        
        if self.nag_wrapper is not None and self.nag_wrapper.nag_enabled:
            self.nag_wrapper.disable_nag()
        
        if should_interpolate:
            logger.info("=" * 80)
            logger.info("Applying frame interpolation...")
            logger.info("=" * 80)
            
            video_tensor = result.get("video")
            if video_tensor is None:
                logger.warning("No video tensor found in result. Skipping interpolation.")
            else:
                interpolator = FrameInterpolator(
                    model_path=frame_interp_config.get("model_path", "models/rife/RIFEv4.26_0921"),
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                
                source_fps = frame_interp_config.get("source_fps", 8.0)
                target_fps = frame_interp_config.get("target_fps", 24.0)
                scale = frame_interp_config.get("scale", 1.0)
                
                logger.info(f"Interpolating: {source_fps} FPS → {target_fps} FPS (scale={scale})")
                
                interpolated_video = interpolator.interpolate(
                    video_tensor=video_tensor,
                    source_fps=source_fps,
                    target_fps=target_fps,
                    scale=scale
                )
                
                interpolator.cleanup()
                
                logger.info("=" * 80)
                logger.info("✓ Frame interpolation complete!")
                logger.info("=" * 80)
                
                if not original_return_tensor and original_save_path:
                    logger.info(f"🎬 Saving interpolated video to {original_save_path} 🎬")
                    from lightx2v.utils.utils import save_to_video
                    save_to_video(interpolated_video, original_save_path, fps=target_fps, method="ffmpeg")
                    logger.info(f"✅ Interpolated video saved successfully to: {original_save_path} ✅")
                    result = {"video": None}
                else:
                    result = {"video": interpolated_video}
        
        if should_interpolate and not original_return_tensor:
            input_info.return_result_tensor = original_return_tensor
            input_info.save_result_path = original_save_path
        
        self.cleanup_after_inference()
        
        logger.info("=" * 80)
        logger.info("✓ Custom distilled inference pipeline complete!")
        logger.info("=" * 80)
        
        return result
