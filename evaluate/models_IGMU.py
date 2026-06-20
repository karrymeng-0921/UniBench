import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel
import os
from tqdm.auto import tqdm
import inspect
from copy import deepcopy
import torchvision.transforms as T
import json
import socket
import scipy.sparse
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from utils import CustomTextEncoder, GEGLU, NeuronRemover, inject_eraser

# ---------- utilities ----------
to_pil = T.ToPILImage()
totensor = T.ToTensor()

# ---------- host & path ----------
HOST_NAME = socket.gethostname()
USER_PATH = "<USER>" if HOST_NAME == "<HOST>" else "<USER>"
ckpt_BASE = f"{USER_PATH}/work/Weights/USD_Doctor/unlearned_ckpt"


class SDAModel(object):
    def __init__(self, unlearn_method, concept, device, num_inference_steps=50,
                 criterion=None, ckpt_path=None, data_type=torch.float16):
        self.num_inference_steps = num_inference_steps
        self.unlearn_method = unlearn_method
        self.concept = concept
        self.device = device
        self.criterion = torch.nn.L1Loss() if criterion == 'l1' else torch.nn.MSELoss()
        self.ckpt_path = ckpt_path
        self.data_type = data_type
        self.strength = 0.8

        self.T = 1000
        self.n_samples = num_inference_steps
        start = self.T // self.n_samples // 2
        self.sampled_t = list(range(start, self.T, self.T // self.n_samples))[:self.n_samples]
        self.generator = torch.Generator(device=self.device)

    # ---------- public API ----------
    def set_(self, seed, guidance_scale):
        self.generator.manual_seed(seed)
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps_, num_inference_steps = self.get_timesteps(self.num_inference_steps, self.strength)

    # ---------- loss helpers ----------
    def get_loss(self, x0, t, encoder_hidden_states, **kwargs):
        noise = self.randn_tensor(x0.shape, generator=self.generator, device=self.device, dtype=x0.dtype)
        noised_latent = x0 * (self.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
            noise * ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
        noised_latent = noised_latent.to(dtype=x0.dtype)
        noise_pred = self.unet(torch.cat([noised_latent] * 2), t,
                               encoder_hidden_states=encoder_hidden_states).sample
        return self.criterion(noise, noise_pred)

    def get_loss_2(self, x0, adv_0, t, encoder_hidden_states, **kwargs):
        # same as get_loss but reserved for future adversarial extension
        return self.get_loss(x0, t, encoder_hidden_states, **kwargs)

    # ---------- load diffusion backbone ----------
    def load_DM(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.data_type, safety_checker=None
        ).to(self.device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.feature_extractor = pipe.feature_extractor
        self.unet_sd = pipe.unet
        del pipe  # release original pipeline

        # ---------- GPU placeholder ----------
        if HOST_NAME != "<HOST>":
            gpu_occ = 0  # 0 for WS | 6 for A100 | 16 for H100
            GPU_models = [[deepcopy(self.unet_sd)] for _ in range(gpu_occ)]

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", num_train_timesteps=1000
        )

        # ---------- load unlearned weights ----------
        if self.unlearn_method == "ORG":
            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method == "SAFECLIP":
            safeclip_text_model = CLIPTextModel.from_pretrained(
                "aimagelab/safeclip_vit-l_14", torch_dtype=self.data_type
            )
            self.text_encoder = safeclip_text_model.to(self.device)
            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method == "UCE21":
            del self.vae, self.tokenizer, self.text_encoder, self.feature_extractor, self.unet_sd
            ldm_stable = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base"
            ).to(self.device)
            self.vae = ldm_stable.vae
            self.tokenizer = ldm_stable.tokenizer
            self.text_encoder = ldm_stable.text_encoder
            self.feature_extractor = ldm_stable.feature_extractor
            self.unet_sd = ldm_stable.unet
            target_ckpt = f"{ckpt_BASE}/<CHECKPOINT>"
            self.unet = deepcopy(self.unet_sd)
            self.unet.load_state_dict(torch.load(target_ckpt, map_location=self.device))

        elif self.unlearn_method in ["ESD", "FMN", "SPM", "RECE", "UCE"]:
            target_ckpt = self.ckpt_path or os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "weight", f"{self.unlearn_method}_{self.concept}_ep2.pt"
            )
            print(f"Loading checkpoint from {target_ckpt}")
            self.unet = deepcopy(self.unet_sd)
            self.unet.load_state_dict(torch.load(target_ckpt, map_location=self.device))

        elif self.unlearn_method == "AdvUnlearn":
            target_ckpt = self.ckpt_path or os.path.join(ckpt_BASE, "<CHECKPOINT>")

            def extract_text_encoder_ckpt(ckpt_path):
                full_ckpt = torch.load(ckpt_path)
                return {k.replace("text_encoder.", ""): v for k, v in full_ckpt.items()
                        if "text_encoder.text_model" in k}

            self.text_encoder.load_state_dict(
                extract_text_encoder_ckpt(target_ckpt), strict=False
            )
            self.text_encoder = self.text_encoder.to(self.device)
            self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
            self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)
            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method == "MACE":
            target_ckpt = self.ckpt_path or f"{ckpt_BASE}/<CHECKPOINT>"
            pipe = StableDiffusionPipeline.from_pretrained(
                target_ckpt, torch_dtype=self.data_type
            ).to(self.device)
            self.vae = pipe.vae
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.unet = pipe.unet

        elif self.unlearn_method == "DoCoPreG":
            target_ckpt = self.ckpt_path or f"{ckpt_BASE}/<CHECKPOINT>"
            st = torch.load(target_ckpt)
            for name, param in self.unet_sd.named_parameters():
                if name in st["unet"]:
                    param.data.copy_(st["unet"][name])
            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method == "ConceptPrune":
            target_ckpt = self.ckpt_path or f"{ckpt_BASE}/<CHECKPOINT>"
            neuron_remover = NeuronRemover(path_expert_indx=self.ckpt_path, T=50,n_layers=16,replace_fn=GEGLU, hook_module='unet'
            )
            pipe = neuron_remover.observe_activation(pipe)
            self.unet = pipe.unet
            self.neuron_remover = neuron_remover 
            return pipe

        elif self.unlearn_method == "Receler":
            target_ckpt = self.ckpt_path or f"{ckpt_BASE}/<CHECKPOINT>"
            eraser_ckpt_path = os.path.join(target_ckpt, "Receler_vangogh_eraser_weights.pt")
            eraser_config_path = os.path.join(target_ckpt, "Receler_vangogh_eraser_config.json")
            with open(eraser_config_path) as f:
                eraser_config = json.load(f)
            inject_eraser(self.unet_sd, torch.load(eraser_ckpt_path, map_location="cpu"), **eraser_config)
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=self.data_type
            ).to(self.device)
            self.unet = deepcopy(pipe.unet)
            for name, module in self.unet.named_modules():
                if "eraser" in name or "adapter" in name:
                    module.to(dtype=self.data_type, device=self.device)

        else:
            raise ValueError(f"Unlearn method {self.unlearn_method} is not supported.")

    # ---------- latent ↔ image ----------
    def latent2_img(self, latents):
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze(0)
        return to_pil(image)

    def randn_tensor(self, shape, generator=None, device=None, dtype=None, layout=None):
        rand_device = device
        batch_size = shape[0]
        layout = layout or torch.strided
        device = device or torch.device("cpu")
        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                       for i in range(batch_size)]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
        return latents

    def prepare_latents(self, image, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)
        latents = self.vae.encode(image).latent_dist.sample(generator)
        return self.vae.config.scaling_factor * latents

    # ---------- prompt encoding ----------
    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance,
                       negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(prompt, padding="max_length",
                                         max_length=self.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask.to(self.device) \
                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask else None
            prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=attention_mask)[0]

        prompt_embeds_dtype = (self.text_encoder.dtype if self.text_encoder is not None
                               else self.unet.dtype if self.unet is not None
                               else prompt_embeds.dtype)
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [""] * batch_size if negative_prompt is None else (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(uncond_tokens, padding="max_length",
                                          max_length=max_length, truncation=True, return_tensors="pt")
            attention_mask = uncond_input.attention_mask.to(self.device) \
                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask else None
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device), attention_mask=attention_mask)[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # ---------- image-to-image ----------
    def get_img_prompt_latent(self, image, prompt, seed=2025):
        prompt_embeds = self._encode_prompt(prompt, 1, self.do_classifier_free_guidance)
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        image = image_processor.preprocess(image)
        org_img_latents = self.prepare_latents(image, prompt_embeds.dtype, self.device, self.generator)
        return org_img_latents, prompt_embeds

    def get_timesteps(self, num_inference_steps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        return timesteps, num_inference_steps - t_start

    def gen_image_ti2i_infer(self, prompt=None, image=None, noise=None, strength=0.8,
                             num_images_per_prompt=1, num_inference_steps=50,
                             init_latents=None, prompt_embeds=None, seed=2025):
        if prompt_embeds is None:
            batch_size = 1 if (prompt is not None and isinstance(prompt, str)) else len(prompt)
            prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt,
                                                 self.do_classifier_free_guidance)
        else:
            batch_size = 1

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        if init_latents is None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            image = image_processor.preprocess(image)
            org_img_latents = self.prepare_latents(image, prompt_embeds.dtype, self.device)
            latents = org_img_latents.clone()
        else:
            latents = init_latents.repeat(batch_size, 1, 1, 1)

        if noise is None:
            noise = self.randn_tensor(latents.shape, generator=self.generator,
                                      device=self.device, dtype=prompt_embeds.dtype)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents = self.scheduler.add_noise(latents, noise, latent_timestep)

        # denoising loop
        for i, t in tqdm(enumerate(timesteps), disable=True):
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image, noise

    # ---------- text-to-image ----------
    def gen_image_t2i_infer(self, prompt=None, num_inference_steps=50, guidance_scale=None,
                            height=512, width=512, seed=2025):
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1
        text_input = self.tokenizer(prompt, padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length",
                                      max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator, device=self.device, dtype=text_embeddings.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(tqdm(self.scheduler.timesteps, disable=True)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
