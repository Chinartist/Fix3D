import sys
from diffusers.video_processor import VideoProcessor
from diffusers import AutoencoderKLWan, WanTransformer3DModel,WanPipeline
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import FlowMatchEulerDiscreteScheduler
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.utils import export_to_video
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
import numpy as np
import random
# from src.vqgan_arch import GANLoss,DiscHead 
from einops import rearrange, repeat
from tqdm import tqdm
class WanC2V(nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__()
        model_id = "/nvme0/public_data/Occupancy/proj/cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae:AutoencoderKLWan = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
        transformer:WanTransformer3DModel = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16)
        with torch.no_grad():
            in_cls = transformer.patch_embedding.__class__ # nn.Conv3d
            old_in_dim = transformer.patch_embedding.in_channels # 16
            new_in_dim = old_in_dim * 2
            new_in = in_cls(
                    new_in_dim,
                    transformer.patch_embedding.out_channels,
                    transformer.patch_embedding.kernel_size,
                    transformer.patch_embedding.stride,
                    transformer.patch_embedding.padding)
            new_in.weight.zero_()
            new_in.bias.zero_()
            new_in.weight[:, :old_in_dim].copy_(transformer.patch_embedding.weight)
            # new_in.weight[:, old_in_dim:].copy_(transformer.patch_embedding.weight)
            new_in.bias.copy_(transformer.patch_embedding.bias)  
            transformer.patch_embedding = new_in
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=torch.float16)
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        scheduler:FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)
        vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
        video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
        num_inference_steps = 1000#scheduler.config.num_train_timesteps
        scheduler.set_timesteps(num_inference_steps, device=vae.device)
    
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")     
            sd = torch.load(pretrained_path, map_location="cpu",weights_only=True)
            transformer.load_state_dict(sd["state_dict_tf"], strict=True)
        self.vae = vae
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler  
        self.video_processor = video_processor
        self.num_inference_steps = num_inference_steps
        
        self.to( dtype=torch.float16)
        self.requires_grad_(False)
    def forward(self, batch):
        x_src = batch["x_src"]#b,c,f,h,w
        x_cond = batch["x_cond"]#b,c,f,h,w
        b, c, f, h, w = x_cond.shape
        x_src = rearrange(x_src, "b c f h w -> (b f) c 1 h w")
        x_cond = rearrange(x_cond, "b c f h w -> (b f) c 1 h w")
        dtype = self.transformer.dtype
        device = self.transformer.device
        self.scheduler.timesteps =self.scheduler.timesteps.to(device)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        timestep = random.choices(self.scheduler.timesteps,k=b)
        timestep = torch.tensor(timestep, device=device)
        step_index = [self.scheduler.index_for_timestep(_t) for _t in timestep]
        step_index = torch.tensor(step_index, device=device, dtype=torch.long)
        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype)
            )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                device, dtype
            )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=batch["prompt"],
            # negative_prompt=batch["negative_prompt"],
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=device,
        )
           
        latents_src = (self.vae.encode(x_src).latent_dist.sample()- latents_mean)* latents_std
        latents_cond = (self.vae.encode(x_cond).latent_dist.sample()- latents_mean)* latents_std
        latents_src = rearrange(latents_src, "(b f) c 1 h w -> b c f h w", b=b, f=f)
        latents_cond = rearrange(latents_cond, "(b f) c 1 h w -> b c f h w", b=b, f=f)
        latents_tgt = latents_src[:,:,0:1]
        noise = torch.randn_like(latents_tgt)
        noise_latents = self.scheduler.scale_noise(sample=latents_tgt, timestep=timestep, noise=noise)# b c 1 h w
        noise_latents_cond = torch.cat([noise_latents, latents_src[:,:,1:]], dim=2) # b c f h w
        noise_latents_cond = torch.cat([noise_latents_cond, latents_cond], dim=1) # b 2c f h w
        noise_pred = self.transformer(
            hidden_states=noise_latents_cond,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]# b 2c f h w
        noise_pred = noise_pred[:,:,0:1] # b c 1 h w
        # target = noise - latents_tgt
        # print(target.sum(),target.mean())
        # fm_loss = F.mse_loss(noise_pred, target, reduction="mean").detach()
        # loss = {"loss_latent": fm_loss}
        loss = {}
        dt = self.scheduler.sigmas[-1]- self.scheduler.sigmas[step_index]
        dt = rearrange(dt, "b -> b 1 1 1 1")
        latents_pred = noise_latents.to(dt)+ dt * noise_pred.to(dt)
        latents_pred = latents_pred.to(dtype)
        loss["loss_latent"] = F.mse_loss(latents_pred, latents_tgt, reduction="mean")
        image_pred = self.vae.decode((latents_pred.detach() /latents_std)+latents_mean,return_dict=False)[0]
        image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
        return image_pred,loss
    @torch.no_grad()
    def infer(self,batch):
        x_src = batch["x_src"]#b,c,f-1,h,w
        x_cond = batch["x_cond"]#b,c,f,h,w
        b, c, f, h, w = x_cond.shape
        x_src = rearrange(x_src, "b c f h w -> (b f) c 1 h w")
        x_cond = rearrange(x_cond, "b c f h w -> (b f) c 1 h w")
        
        dtype = self.transformer.dtype
        device = self.transformer.device
        self.scheduler.timesteps =self.scheduler.timesteps.to(device)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype)
            )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                device, dtype
            )
        prompt_embeds, _ = self.encode_prompt(
            prompt=batch["prompt"],
            negative_prompt=None,
            do_classifier_free_guidance=False,
            max_sequence_length=512,
            device=device,
        )

        latents_src = (self.vae.encode(x_src).latent_dist.sample()- latents_mean)* latents_std
        latents_cond = (self.vae.encode(x_cond).latent_dist.sample()- latents_mean)* latents_std
        latents_src = rearrange(latents_src, "(b f) c 1 h w -> b c f h w", b=b, f=f-1)
        latents_cond = rearrange(latents_cond, "(b f) c 1 h w -> b c f h w", b=b, f=f)
        noise = torch.randn_like(latents_src[:,:,0:1]) # b c 1 h w

        noise_latents = noise
        for i,t in enumerate(tqdm(self.scheduler.timesteps)):
           
            timestep = t
            step_index = self.scheduler.index_for_timestep(timestep)
            
            noise_latents_cond = torch.cat([noise_latents, latents_src], dim=2) # b c f h w
            noise_latents_cond = torch.cat([noise_latents_cond, latents_cond], dim=1) # b 2c f h w
            noise_pred = self.transformer(
                hidden_states=noise_latents_cond,
                timestep=timestep.expand(b),
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]# b c f h w
            noise_pred = noise_pred[:,:,0:1] # b c 1 h w
            dt = self.scheduler.sigmas[step_index+1]- self.scheduler.sigmas[step_index]

            noise_latents = noise_latents.to(dt) + dt * noise_pred.to(dt)
            noise_latents = noise_latents.to(dtype)
        latents_pred = noise_latents
        image_pred = self.vae.decode((latents_pred.detach() /latents_std)+latents_mean,return_dict=False)[0]
        image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
        return image_pred
            
    def set_train(self,):
        
        self.train()
        self.requires_grad_(False)
        self.transformer.requires_grad_(True)
  
    def set_eval(self,):
        self.eval()
        self.requires_grad_(False)
    def save_model(self, save_path: str):
        sd = {}
        sd["state_dict_tf"] = self.transformer.state_dict()
        torch.save(sd, save_path)
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

if __name__ == "__main__":
    dtype = torch.bfloat16
    model = WanC2V().cuda().to(dtype=dtype)
    model.set_train()
    batch = {
        "x_src": torch.randn(2, 3, 5,256, 256).to(dtype=dtype).cuda(),
        "x_cond": torch.randn(2, 3, 5,256, 256).to(dtype=dtype).cuda(),
        "prompt": ["a cat", "a dog"],
        "negative_prompt": ["bad cat", "bad dog"]
    }
    image_pred, loss = model(batch)
    print(image_pred.shape, loss)
    batch = {
        "x_src": torch.randn(2, 3, 4,256, 256).to(dtype=dtype).cuda(),
        "x_cond": torch.randn(2, 3, 5,256, 256).to(dtype=dtype).cuda(),
        "prompt": ["a cat", "a dog"],
        "negative_prompt": ["bad cat", "bad dog"]
    }
    image_pred = model.infer(batch)
    print(image_pred.shape)
