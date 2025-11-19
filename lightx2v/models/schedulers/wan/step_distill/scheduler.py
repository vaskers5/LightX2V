import math
from typing import List, Union

import torch
from loguru import logger

from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanStepDistillScheduler(WanScheduler):
    def __init__(self, config):
        requested_infer_steps = config.get("infer_steps")
        denoising_step_list = self._build_denoising_schedule(config)

        config["denoising_step_list"] = denoising_step_list
        config["infer_steps"] = len(denoising_step_list)
        if requested_infer_steps is not None and requested_infer_steps != config["infer_steps"]:
            logger.warning(
                "infer_steps=%s does not match len(denoising_step_list)=%s, overriding to %s",
                requested_infer_steps,
                len(denoising_step_list),
                len(denoising_step_list),
            )

        super().__init__(config)
        self.denoising_step_list = denoising_step_list
        self.infer_steps = len(self.denoising_step_list)
        self.sample_shift = self.config["sample_shift"]

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    @staticmethod
    def _build_denoising_schedule(config) -> List[int]:
        high_steps = config.get("denoising_step_list_high")
        low_steps = config.get("denoising_step_list_low")

        if high_steps or low_steps:
            if not high_steps or not low_steps:
                raise ValueError("Both `denoising_step_list_high` and `denoising_step_list_low` must be provided together")
            combined_steps = list(high_steps) + list(low_steps)
            config.setdefault("boundary_step_index", len(high_steps))
        else:
            base_list = config.get("denoising_step_list")
            if not base_list:
                raise ValueError("`denoising_step_list` must be provided for WanStepDistillScheduler")
            combined_steps = list(base_list)

        max_timestep = config.get("num_train_timesteps", 1000)
        normalized_steps = [int(step) for step in combined_steps]
        if len(normalized_steps) < 1:
            raise ValueError("`denoising_step_list` must contain at least one timestep")

        if any(normalized_steps[idx] <= normalized_steps[idx + 1] for idx in range(len(normalized_steps) - 1)):
            raise ValueError("`denoising_step_list` must be strictly decreasing")

        if any(step > max_timestep or step < 0 for step in normalized_steps):
            raise ValueError(f"`denoising_step_list` values must be within [0, {max_timestep}]")

        return normalized_steps

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)
        self.set_denoising_timesteps(device=self.device)

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def reset(self, seed, latent_shape, step_index=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)

    def add_noise(self, original_samples, noise, sigma):
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.infer_steps - 1:
            sigma = self.sigmas[self.step_index + 1].item()
            noise = torch.randn(noisy_image_or_video.shape, dtype=torch.float32, device=self.device, generator=self.generator)
            noisy_image_or_video = self.add_noise(noisy_image_or_video, noise=noise, sigma=self.sigmas[self.step_index + 1].item())
        self.latents = noisy_image_or_video.to(self.latents.dtype)


class Wan22StepDistillScheduler(WanStepDistillScheduler):
    def __init__(self, config):
        super().__init__(config)
        max_index = self.infer_steps - 1
        requested_boundary = config.get("boundary_step_index", max_index)
        if requested_boundary > max_index or requested_boundary < 0:
            logger.warning(
                f"boundary_step_index={requested_boundary} is out of range for {self.infer_steps} steps, clamping to [0, {max_index}]"
            )
            requested_boundary = min(max(requested_boundary, 0), max_index)

        self.boundary_step_index = requested_boundary

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        super().set_denoising_timesteps(device)
        self.sigma_bound = self.sigmas[self.boundary_step_index].item()

    def calculate_alpha_beta_high(self, sigma):
        alpha = (1 - sigma) / (1 - self.sigma_bound)
        beta = math.sqrt(sigma**2 - (alpha * self.sigma_bound) ** 2)
        return alpha, beta

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - flow_pred * sigma
        # self.latent: x_t
        if self.step_index < self.infer_steps - 1:
            sigma_n = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = noisy_image_or_video + flow_pred * sigma_n

        self.latents = noisy_image_or_video.to(self.latents.dtype)
