"""CASI dual-branch generation using the standard diffusers model components."""

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


class CASIPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self, prompt_embeds_global, prompt_embeds_defect,
        num_inference_steps=100, guidance_scale=2.5,
        generator=None, height=512, width=512,
    ):
        if prompt_embeds_global.shape != prompt_embeds_defect.shape:
            raise ValueError("Global/defect prompt embeddings must have the same shape")
        if height <= 0 or width <= 0 or height % 8 or width % 8:
            raise ValueError("Image dimensions must be positive multiples of 8")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        device = self._execution_device
        dtype = self.unet.dtype
        batch_size = prompt_embeds_global.shape[0]
        embeddings = torch.stack((prompt_embeds_global, prompt_embeds_defect), dim=1).flatten(0, 1)
        embeddings = embeddings.to(device=device, dtype=dtype)
        do_guidance = guidance_scale > 1.0
        if do_guidance:
            ids = self.tokenizer(
                [""] * batch_size, padding="max_length", truncation=True,
                max_length=self.tokenizer.model_max_length, return_tensors="pt",
            ).input_ids.to(device)
            # CODD has no active token masks for the unconditional prompt.
            unconditional = self.text_encoder(ids)[0].to(dtype=dtype).repeat_interleave(2, dim=0)
            embeddings = torch.cat((unconditional, embeddings))

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self.prepare_latents(
            batch_size * 2, self.unet.config.in_channels, height, width,
            dtype, device, generator, None,
        )
        extra_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            model_input = torch.cat((latents, latents)) if do_guidance else latents
            model_input = self.scheduler.scale_model_input(model_input, timestep)
            predicted = self.unet(model_input, timestep, encoder_hidden_states=embeddings).sample
            if do_guidance:
                unconditional, conditional = predicted.chunk(2)
                predicted = unconditional + guidance_scale * (conditional - unconditional)
            latents = self.scheduler.step(predicted, timestep, latents, **extra_kwargs).prev_sample

        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        images = self.image_processor.postprocess(images, output_type="pil")
        # Return [global_0, defect_0, global_1, defect_1, ...], as in the source.
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
