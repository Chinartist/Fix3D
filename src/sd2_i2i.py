import os
import requests
import sys


p = "src/"

sys.path.append(p)

import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel,CLIPVisionModelWithProjection,CLIPVisionModel
from diffusers import DDPMScheduler,DDIMScheduler,EulerDiscreteScheduler,PNDMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from torch.nn import functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps
from typing import Optional, List, Tuple, Union
import inspect
from torch import nn
import random
from copy import deepcopy

class SD2I2I(torch.nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__()
        model_id = "/nvme0/public_data/Occupancy/proj/cache/stabilityai/sd-turbo"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.scheduler =EulerDiscreteScheduler().from_pretrained(model_id, subfolder="scheduler")
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, 1, "cuda", None)
        assert len(timesteps) == 1
        self.timesteps = torch.tensor(timesteps).cuda()
        print(f"Num inference steps actual: {num_inference_steps}")
        print(f"First timestep: {timesteps[0]}")
        vae:AutoencoderKL = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        unet:UNet2DConditionModel = UNet2DConditionModel.from_pretrained("/nvme0/public_data/Occupancy/proj/cache/nvidia/difix", subfolder="unet")
        unet.enable_xformers_memory_efficient_attention()
        if pretrained_path is not None and pretrained_path != "":
            print("Loading pretrained model from path:", pretrained_path)
            sd = torch.load(pretrained_path, map_location="cpu")
            unet.load_state_dict(sd["state_dict_unet"])
        self.unet, self.vae = unet, vae
        self.requires_grad_(False)
    
    def forward(self, batch):
        # either the prompt or the prompt_tokens should be provided
        text_prompt_embeds = self.encode_prompt(batch["prompt"])
        
        sample = self.vae.encode(batch["x_tgt"]).latent_dist.sample() * self.vae.config.scaling_factor
        cond = self.vae.encode(batch["x_src"]).latent_dist.sample() * self.vae.config.scaling_factor
        
        latent_model_input = cond
        latent_model_input= self.scheduler.scale_model_input(latent_model_input, self.timesteps[0])
        noise_pred = self.unet(latent_model_input, self.timesteps[0], encoder_hidden_states=text_prompt_embeds, return_dict=False,
            )[0]
        sample_pred = self.scheduler.step(noise_pred,  self.timesteps[0], latent_model_input,generator=None, return_dict=False)[0]
        self.scheduler._init_step_index(self.timesteps[0])
        image_pred = (self.vae.decode(sample_pred / self.vae.config.scaling_factor, return_dict=False, generator=None)[
                            0
                        ])
        image_pred = image_pred+(image_pred.clamp(-1, 1)-image_pred).detach()
        loss ={}
        loss["loss_latent"]= F.mse_loss(sample_pred.detach(), sample)
        
        return image_pred,sample_pred,sample,loss
    @torch.no_grad()
    def infer(self, batch):
        # either the prompt or the prompt_tokens should be provided
        text_prompt_embeds = self.encode_prompt(batch["prompt"])
        
        # sample = self.vae.encode(batch["x_tgt"]).latent_dist.sample() * self.vae.config.scaling_factor
        cond = self.vae.encode(batch["x_src"]).latent_dist.sample() * self.vae.config.scaling_factor
        
        latent_model_input = cond
        latent_model_input= self.scheduler.scale_model_input(latent_model_input, self.timesteps[0])
        noise_pred = self.unet(latent_model_input, self.timesteps[0], encoder_hidden_states=text_prompt_embeds, return_dict=False,
            )[0]
        sample_pred = self.scheduler.step(noise_pred,  self.timesteps[0], latent_model_input,generator=None, return_dict=False)[0]
        self.scheduler._init_step_index(self.timesteps[0])
        image_pred = (self.vae.decode(sample_pred / self.vae.config.scaling_factor, return_dict=False, generator=None)[
                            0
                        ]).clamp(-1, 1)
        # loss ={}
        # loss["loss_latent"]= F.mse_loss(sample_pred.detach(), sample)
        
        return image_pred
    def set_eval(self):
        self.eval()
        self.requires_grad_(False)
    def set_train(self):
        self.train()
        self.unet.requires_grad_(True)
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds
    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items()}
        torch.save(sd, outf)



class SD2Reg(torch.nn.Module):
    def __init__(self, ):
        super().__init__() 
        model_id = "stabilityai/stable-diffusion"
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
        self.noise_scheduler:PNDMScheduler = PNDMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")

        self.unet_fix = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
        lora_rank_unet=4
        target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
            target_modules=target_modules_unet
        )
        unet= UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
        unet.add_adapter(unet_lora_config)
        self.unet_update =unet
        self.unet_fix.enable_xformers_memory_efficient_attention()
        self.unet_update.enable_xformers_memory_efficient_attention()

        self.requires_grad_(False)
    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds
    def diff_loss(self, latents, prompt_embeds):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input,
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds,
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss