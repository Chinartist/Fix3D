import sys
p = "src/"
sys.path.append(p)
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
from vqgan_arch import GANLoss,DiscHead
class WanV2V(nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__()
        model_id = "/nvme0/public_data/Occupancy/proj/cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae:AutoencoderKLWan = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
        transformer:WanTransformer3DModel = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16)
        # with torch.no_grad():
        #     in_cls = transformer.patch_embedding.__class__ # nn.Conv3d
        #     old_in_dim = transformer.patch_embedding.in_channels # 16
        #     new_in_dim = old_in_dim * 2
        #     new_in = in_cls(
        #             new_in_dim,
        #             transformer.patch_embedding.out_channels,
        #             transformer.patch_embedding.kernel_size,
        #             transformer.patch_embedding.stride,
        #             transformer.patch_embedding.padding)
        #     new_in.weight.zero_()
        #     new_in.bias.zero_()
        #     new_in.weight[:, :old_in_dim].copy_(transformer.patch_embedding.weight)
        #     new_in.bias.copy_(transformer.patch_embedding.bias)  
        #     transformer.patch_embedding = new_in
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=torch.float16)
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        scheduler:FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)
        vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
        video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
        num_inference_steps = 1#scheduler.config.num_train_timesteps
        scheduler.set_timesteps(num_inference_steps, device=vae.device)
        # if pretrained_path is None:
        #     target_modules = set()
        #     for name, module in transformer.named_modules():
        #         if isinstance(module, torch.nn.Linear):
        #             target_modules.add(name)
        #     transformer_lora_config = LoraConfig(
        #     r=512,
        #     target_modules=target_modules,
        #     init_lora_weights=True,)
        #     transformer.add_adapter(transformer_lora_config)
        #     lora_config_tf = dict(
        #         r =transformer_lora_config.r,
        #         target_modules = transformer_lora_config.target_modules
        #     )
        # else:
        #     print(f"Loading pretrained lora weights from {pretrained_path}")
        #     sd = torch.load(pretrained_path, map_location="cpu")
        #     transformer_lora_config = LoraConfig(
        #         r=sd['lora_config_tf']['r'],
        #         target_modules=sd['lora_config_tf']['target_modules'],
        #     )
        #     transformer.add_adapter(transformer_lora_config)
        #     _sd_tf = transformer.state_dict()
        #     for k, v in sd['state_dict_tf'].items():
        #         if k in _sd_tf:
        #             _sd_tf[k] = v
        #     transformer.load_state_dict(_sd_tf, strict=True)
        #     lora_config_tf = sd['lora_config_tf']
        # self.lora_config_tf = lora_config_tf
        self.vae = vae
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.video_processor = video_processor
        self.num_inference_steps = num_inference_steps
        
        self.to(torch.float16)
        self.requires_grad_(False)
    def forward(self, batch):
        x_src = batch["x_src"][:,:,None]
        x_tgt = batch["x_tgt"][:,:,None]
        dtype = self.transformer.dtype
        device = self.transformer.device
        self.scheduler.timesteps =self.scheduler.timesteps.to(device)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        timestep = random.choices(self.scheduler.timesteps,k=x_src.shape[0])
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
            negative_prompt=batch["negative_prompt"],
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=device,
        )
        
        cond = (self.vae.encode(x_src).latent_dist.sample()- latents_mean)* latents_std
        sample = (self.vae.encode(x_tgt).latent_dist.sample()- latents_mean)* latents_std
        noise = torch.randn_like(cond)
        noise_latents = self.scheduler.scale_noise(sample=sample, timestep=timestep, noise=cond)
        noise_latents_cond =cond# torch.cat([noise_latents,cond], dim=1)
        noise_pred = self.transformer(
            hidden_states=noise_latents_cond,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        # target = noise - sample
        # print(target.sum(),target.mean())
        # fm_loss = F.mse_loss(noise_pred, target, reduction="mean").detach()
        # loss = {"loss_latent": fm_loss}
        loss = {}
        dt = self.scheduler.sigmas[-1]- self.scheduler.sigmas[step_index]
        dt = dt.reshape(-1, 1, 1, 1, 1)
        latents_pred = noise_latents+ dt * noise_pred
        loss["loss_latent"] = F.mse_loss(latents_pred, sample, reduction="mean")
        image_pred = self.vae.decode((latents_pred /latents_std)+latents_mean,return_dict=False)[0]
        image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
        return image_pred,cond,latents_pred,sample,loss,prompt_embeds, negative_prompt_embeds,None
    @torch.no_grad()
    def infer(self,batch):
        x_src = batch["x_src"][:,:,None]
        dtype = self.transformer.dtype
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

        cond = (self.vae.encode(x_src).latent_dist.sample()- latents_mean)* latents_std
        noise = torch.randn_like(cond)
        noise_latents = noise
        for i,t in enumerate(self.scheduler.timesteps):
           
            timestep = t
            step_index = self.scheduler.index_for_timestep(timestep)
            
            noise_latents_cond = torch.cat([noise_latents,cond], dim=1)
            noise_pred = self.transformer(
                hidden_states=noise_latents_cond,
                timestep=timestep.expand(noise_latents_cond.shape[0]),
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            dt = self.scheduler.sigmas[step_index+1]- self.scheduler.sigmas[step_index]
            dt = dt.reshape(-1, 1, 1, 1, 1)
            noise_latents = noise_latents + dt * noise_pred
        latents_pred = noise_latents
        image_pred = self.vae.decode((latents_pred.detach() /latents_std)+latents_mean,return_dict=False)[0]
        image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
        return image_pred
            
    def set_train(self,):
        
        self.train()
        self.requires_grad_(False)
        self.transformer.requires_grad_(True)
        # for n, _p in self.transformer.named_parameters(): 
        #     if "lora" in n or "patch_embedding" in n or "proj_out" in n:
        #         _p.requires_grad = True
    def set_eval(self,):
        self.eval()
        self.requires_grad_(False)
    def save_model(self, save_path: str):
        sd = {}
        # sd["state_dict_tf"] ={ k: v for k, v in self.transformer.state_dict().items() if "lora" in k or "patch_embedding" in k or "proj_out" in k}
        sd["state_dict_tf"] = self.transformer.state_dict()
        # sd["lora_config_tf"] = self.lora_config_tf
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


def wan_transformer_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0



        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        hook = self.hooks
        # 4. Transformer blocks
        hook_states = []
        for i,block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if i in hook:
                hook_states.append(hidden_states)
        return hook_states
from diffusers.models.transformers.transformer_wan import FP32LayerNorm

        
class WanV2VPercepGan(nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__()
        model_id = "/nvme0/public_data/Occupancy/proj/cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16
        )
        scheduler:FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)
        
        max_hook = 30
        gan_hooks = list(range(8, max_hook,4))

        transformer.blocks = nn.ModuleList([
            transformer.blocks[i] for i in range(max_hook)
        ])
        transformer.forward = wan_transformer_forward.__get__(transformer, WanTransformer3DModel)


        heads = nn.ModuleList()
        inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        for i in range(len(gan_hooks)):
            heads.append(nn.Sequential(
                FP32LayerNorm(inner_dim),
                nn.Linear(inner_dim, inner_dim, bias=False),
                nn.GELU(),
                nn.Linear(inner_dim, 1, bias=False),)
            )
        if pretrained_path is None:
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            transformer_lora_config = LoraConfig(
            r=128,
            target_modules=target_modules,
            init_lora_weights=True,)
            transformer.add_adapter(transformer_lora_config)
            lora_config_tf = dict(
                r =transformer_lora_config.r,
                target_modules = transformer_lora_config.target_modules
            )
            pass
        else:

            print(f"WanV2VPercepGan: Loading pretrained  weights from {pretrained_path}")
            sd = torch.load(pretrained_path, map_location="cpu")
            heads.load_state_dict(sd['heads'])
            transformer_lora_config = LoraConfig(
                r=sd['lora_config_tf']['r'],
                target_modules=sd['lora_config_tf']['target_modules'],
            )
            transformer.add_adapter(transformer_lora_config)
            _sd_tf = transformer.state_dict()
            for k, v in sd['state_dict_tf'].items():
                if k in _sd_tf:
                    _sd_tf[k] = v
            transformer.load_state_dict(_sd_tf, strict=True)
            lora_config_tf = sd['lora_config_tf']
        self.lora_config_tf = lora_config_tf
        self.percep_loss_fn = F.mse_loss
        self.gan_loss_fn = GANLoss("hinge")
        self.max_timestep = 10
        self.gan_hooks = gan_hooks
        print("GAN hooks:", self.gan_hooks)
        self.transformer = transformer
        self.scheduler = scheduler
        self.heads = heads
        self.requires_grad_(False)
        for n, _p in self.transformer.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.heads.requires_grad_(True)
        self.float16()
      
    def forward(self, latents_pred=None,latents_gt=None, prompt_embeds=None,for_real=False,for_disc=False):
        if 0:
            for n, _p in self.transformer.named_parameters():
                if "lora" in n:
                    _p.requires_grad = False
            self.heads.requires_grad_(False)
            self.transformer.hooks = self.gan_hooks
            noise = torch.randn_like(latents_pred)
            self.scheduler.timesteps = self.scheduler.timesteps.to(latents_pred.device,)
            timestep = random.choices(self.scheduler.timesteps[self.scheduler.timesteps <self.max_timestep], k=latents_pred.shape[0])
            timestep = torch.tensor(timestep, device=latents_pred.device, )
            noise_latents_pred = self.scheduler.scale_noise(sample=latents_pred, timestep=timestep, noise=noise)
            noise_latents_gt = self.scheduler.scale_noise(sample=latents_gt, timestep=timestep, noise=noise)
            noise_latents = torch.cat([noise_latents_pred, noise_latents_gt], dim=0)
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
          
            hook_states = self.transformer(
                hidden_states=noise_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )
            
            hook_states_pred = []
            hook_states_gt = []
            for i, h in enumerate(self.gan_hooks):
                head = self.heads[i]
                hook_states_pred.append(head(hook_states[i][:latents_pred.shape[0]]))
                hook_states_gt.append(head(hook_states[i][latents_pred.shape[0]:]))
            hook_states_pred = torch.stack(hook_states_pred, dim=1)
            hook_states_gt = torch.stack(hook_states_gt, dim=1)
            percep_loss = self.percep_loss_fn(
                hook_states_pred, hook_states_gt, reduction="mean"
            )
            return percep_loss
        else:
            self.transformer.hooks = self.gan_hooks
            if for_disc:
                for n, _p in self.transformer.named_parameters():
                    if "lora" in n:
                        _p.requires_grad = True
                self.heads.requires_grad_(True)
                if for_real:
                    noise = torch.randn_like(latents_gt)
                    self.scheduler.timesteps = self.scheduler.timesteps.to(latents_gt.device)
                    timestep = random.choices(self.scheduler.timesteps[self.scheduler.timesteps <self.max_timestep], k=latents_gt.shape[0])
                    timestep = torch.tensor(timestep, device=latents_gt.device, )
                    noise_latents = self.scheduler.scale_noise(sample=latents_gt, timestep=timestep, noise=noise)
                else:
                    noise = torch.randn_like(latents_pred)
                    self.scheduler.timesteps = self.scheduler.timesteps.to(latents_pred.device,)
                    timestep = random.choices(self.scheduler.timesteps[self.scheduler.timesteps <self.max_timestep],k=latents_pred.shape[0])
                    timestep = torch.tensor(timestep, device=latents_pred.device, )
                    noise_latents = self.scheduler.scale_noise(sample=latents_pred, timestep=timestep, noise=noise)
            else:
                for n, _p in self.transformer.named_parameters():
                    if "lora" in n:
                        _p.requires_grad = False
                self.heads.requires_grad_(False)
                noise = torch.randn_like(latents_pred)
                self.scheduler.timesteps = self.scheduler.timesteps.to(latents_pred.device,)
                timestep = random.choices(self.scheduler.timesteps[self.scheduler.timesteps <self.max_timestep], k=latents_pred.shape[0])
                timestep = torch.tensor(timestep, device=latents_pred.device, )
                noise_latents = self.scheduler.scale_noise(sample=latents_pred, timestep=timestep, noise=noise)
          
            hook_states = self.transformer(
                hidden_states=noise_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )
            gan_loss = 0.0
            for i, h in enumerate(self.gan_hooks):
                hook_states_i = hook_states[i]
                head = self.heads[i]
                gan_loss += self.gan_loss_fn(head(hook_states_i), for_real=for_real,for_disc=for_disc)
            gan_loss = gan_loss / len(self.gan_hooks)
            return gan_loss
    def save_model(self, save_path: str):
        sd = {}
        try:
            sd["state_dict_tf"] = {k: v for k, v in self.transformer.state_dict().items() if "lora" in k}
            sd["lora_config_tf"] = self.lora_config_tf
        except:
            pass
        sd["heads"] = self.heads.state_dict()
        torch.save(sd, save_path)

class WanV2VReg(nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__()
        model_id = "/nvme0/public_data/Occupancy/proj/cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
   
        scheduler:FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", torch_dtype=torch.float16)
        
        self.transformer_real =  WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16
        )
        self.transformer_fake = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float16
        )
        target_modules = set()
        for name, module in self.transformer_fake .named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.add(name)
        transformer_lora_config = LoraConfig(
        r=4,
        target_modules=target_modules,
        init_lora_weights=True,)
        
        self.transformer_fake.add_adapter(transformer_lora_config)
        self.transformer_real.add_adapter(transformer_lora_config)
        self.scheduler = scheduler
        self.min_dm_step = int(self.scheduler.config.num_train_timesteps * 0.005)
        self.max_dm_step = int(self.scheduler.config.num_train_timesteps * 0.4)
        self.guidance_scale = 5.0

        self.requires_grad_(False)
        for n, _p in self.transformer_fake.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for n, _p in self.transformer_real.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        print(f"Distribution matching step range: {self.min_dm_step} - {self.max_dm_step}")
    def update(self,latents_pred,latents_gt, prompt_embeds):
        noise = torch.randn_like(latents_pred)
        self.scheduler.timesteps = self.scheduler.timesteps.to(latents_pred.device,)
        timestep = random.choices(self.scheduler.timesteps[( self.scheduler.timesteps > self.min_dm_step) & (self.scheduler.timesteps < self.max_dm_step)], k=latents_pred.shape[0])
        timestep = torch.tensor(timestep, device=latents_pred.device, )
        noise_latents = self.scheduler.scale_noise(sample=latents_pred, timestep=timestep, noise=noise)
        noise_pred_fake = self.transformer_fake(
            hidden_states=noise_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        target = noise-latents_pred
        loss = F.mse_loss(noise_pred_fake, target, reduction="mean")

        noise = torch.randn_like(latents_gt)
        timestep = random.choices(self.scheduler.timesteps[( self.scheduler.timesteps > self.min_dm_step) & (self.scheduler.timesteps < self.max_dm_step)], k=latents_gt.shape[0])
        timestep = torch.tensor(timestep, device=latents_gt.device, )
        noise_latents = self.scheduler.scale_noise(sample=latents_gt, timestep=timestep, noise=noise)
        noise_pred_real = self.transformer_real(
            hidden_states=noise_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        target = noise-latents_gt
        loss += F.mse_loss(noise_pred_real, target, reduction="mean")
        return loss
    def forward(self, latents_pred=None, prompt_embeds=None, negative_prompt_embeds=None):
        noise = torch.randn_like(latents_pred)
        self.scheduler.timesteps = self.scheduler.timesteps.to(latents_pred.device,)
        self.scheduler.sigmas = self.scheduler.sigmas.to(latents_pred.device,)
        timestep = random.choices(self.scheduler.timesteps[( self.scheduler.timesteps > self.min_dm_step) & (self.scheduler.timesteps < self.max_dm_step)], k=latents_pred.shape[0])
        timestep = torch.tensor(timestep, device=latents_pred.device, )
        step_index = [self.scheduler.index_for_timestep(_t) for _t in timestep]
        step_index = torch.tensor(step_index, device=latents_pred.device, dtype=torch.long)
        noise_latents = self.scheduler.scale_noise(sample=latents_pred, timestep=timestep, noise=noise)
           
        with torch.no_grad():
            noise_pred_fake = self.transformer_fake(
                hidden_states=noise_latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            dt = self.scheduler.sigmas[-1]- self.scheduler.sigmas[step_index]
            dt = dt.reshape(-1, 1, 1, 1, 1)
            latents_pred_fake = noise_latents + dt * noise_pred_fake



            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
            noise_latents_input = torch.cat([noise_latents, noise_latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)
            noise_pred = self.transformer_real(
                hidden_states=noise_latents_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred,noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)
            dt = self.scheduler.sigmas[-1]- self.scheduler.sigmas[step_index]
            dt = dt.reshape(-1, 1, 1, 1, 1)
            latents_pred_real = noise_latents+ dt * noise_pred
        weighting_factor = torch.abs(latents_pred - latents_pred_real).mean(dim=[1, 2, 3,4], keepdim=True)
        grad = (latents_pred_fake - latents_pred_real) / weighting_factor
        loss = F.mse_loss(latents_pred, (latents_pred - grad).detach())
        return loss