from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast
from diffusers import HunyuanVideoPipeline,AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
import torch
from diffusers.utils import export_to_video
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps,DEFAULT_PROMPT_TEMPLATE
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import nn
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import numpy as np
import random
class HunyuanV2V(nn.Module):
    def __init__(self,pretrained_path=None):
        super().__init__()
        # pipe:HunyuanVideoPipeline = HunyuanVideoPipeline.from_pretrained(
        #     "/nvme0/public_data/Occupancy/proj/cache/hunyuanvideo-community/HunyuanVideo",torch_dtype=torch.bfloat16,)
        # print("Loading HunyuanVideoPipeline...")
        # self.pipe = pipe
        model_path = "/nvme0/public_data/Occupancy/proj/cache/hunyuanvideo-community/HunyuanVideo"
        self.text_encoder = LlamaModel.from_pretrained(
            model_path,subfolder="text_encoder", torch_dtype=torch.bfloat16,
        )
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            model_path,subfolder="text_encoder_2", torch_dtype=torch.bfloat16,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            model_path,subfolder="tokenizer", torch_dtype=torch.bfloat16,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_path,subfolder="tokenizer_2", torch_dtype=torch.bfloat16,
        )
        self.scheduler:FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler", torch_dtype=torch.bfloat16,
        )
        self.vae:AutoencoderKLHunyuanVideo = AutoencoderKLHunyuanVideo.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16,
        )
        self.transformer:HunyuanVideoTransformer3DModel = HunyuanVideoTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        num_inference_steps = 10
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, "cuda", sigmas=sigmas)
        self.step_indexs = np.arange(num_inference_steps, dtype=np.int64)
        self.sigmas = self.scheduler.sigmas
        self.timesteps = timesteps
        print(f"Using {num_inference_steps} inference steps with step_indexs {self.step_indexs} sigmas: {self.sigmas} and timesteps: {self.timesteps}")
        self.guidance_scale = 6.0
        if pretrained_path is None:
            target_modules = set()
            for name, module in self.transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            transformer_lora_config = LoraConfig(
            r=128,
            lora_alpha=128,
            target_modules=target_modules,
            init_lora_weights=True,)
            self.transformer.add_adapter(transformer_lora_config)
            self.lora_config_tf = dict(
                r =transformer_lora_config.r,
                lora_alpha = transformer_lora_config.lora_alpha,
                target_modules = transformer_lora_config.target_modules
            )
        else:
            sd = torch.load(pretrained_path, map_location="cpu")
            transformer_lora_config = LoraConfig(
                r=sd['lora_config_tf']['r'],
                lora_alpha=sd['lora_config_tf']['lora_alpha'],
                target_modules=sd['lora_config_tf']['target_modules'],
            )
            self.transformer.add_adapter(transformer_lora_config)
            _sd_tf = self.transformer.state_dict()
            for k, v in sd['state_dict_tf'].items():
                if k in _sd_tf:
                    _sd_tf[k] = v
            self.transformer.load_state_dict(_sd_tf, strict=True)
            self.lora_config_tf = sd['lora_config_tf']
        self.bfloat16()
    def forward(self, batch):
        batch["x_src"] = batch["x_src"][:,:,None]
        batch["x_tgt"] = batch["x_tgt"][:,:,None]
        transformer_dtype = self.transformer.dtype
        device = self.transformer.device
        step_index = random.choice(self.step_indexs)
        timestep = self.timesteps[step_index]
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=batch['prompt'],
            device=device,
        )
        noise = self.vae.encode(batch["x_src"]).latent_dist.sample()*self.vae.config.scaling_factor
        sample = self.vae.encode(batch["x_tgt"]).latent_dist.sample()*self.vae.config.scaling_factor

        sigma = self.sigmas[step_index]
        noise_latents = self.scheduler.scale_noise(sample=sample, timestep=timestep[None], noise=noise)
        guidance = torch.tensor([self.guidance_scale] * noise_latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        timestep = timestep.expand(noise_latents.shape[0]).to(noise_latents.dtype)
        noise_pred = self.transformer(
                    hidden_states=noise_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
        target = noise - sample
        fm_loss = torch.nn.functional.mse_loss(noise_pred, target, reduction="mean")
        loss = {"fm_loss": fm_loss}

        dt = self.scheduler.sigmas[-1]- self.scheduler.sigmas[step_index]
        sample_pred = noise_latents+ dt * noise_pred
        x0_pred = self.vae.decode(sample_pred.detach() / self.vae.config.scaling_factor,return_dict=False)[0]
        x0_pred  = x0_pred[:,:,0].clamp(-1.0, 1.0)
        return x0_pred,sample_pred,sample,loss,None
    @torch.no_grad()
    def infer(self,batch):
        batch["x_src"] = batch["x_src"][:,:,None]
        transformer_dtype = self.transformer.dtype
        device = self.transformer.device
        noise_latents = self.vae.encode(batch["x_src"]).latent_dist.sample()*self.vae.config.scaling_factor
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt=batch['prompt'],
                device=device,
            )
        for step_index in self.step_indexs:
            timestep = self.timesteps[step_index]
            
            sigma = self.sigmas[step_index]
            guidance = torch.tensor([self.guidance_scale] * noise_latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

            timestep = timestep.expand(noise_latents.shape[0]).to(noise_latents.dtype)
            noise_pred = self.transformer(
                        hidden_states=noise_latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
            dt = self.scheduler.sigmas[step_index+1]- self.scheduler.sigmas[step_index]
            noise_latents = noise_latents + dt * noise_pred
        x0_pred = self.vae.decode(noise_latents.detach() / self.vae.config.scaling_factor,return_dict=False)[0]
        x0_pred  = x0_pred[:,:,0].clamp(-1.0, 1.0)
        return x0_pred
    def set_train(self,):
        self.train()
        self.requires_grad_(False)
        for n, _p in self.transformer.named_parameters(): 
            if "lora" in n:
                _p.requires_grad = True
    def set_eval(self,):
        self.eval()
        self.requires_grad_(False)
    def save_model(self, path):
        sd = {}
        sd["state_dict_tf"] ={ k: v for k, v in self.transformer.state_dict().items() if "lora" in k}
        sd["lora_config_tf"] = self.lora_config_tf
        torch.save(sd, path)
    def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device 
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_videos_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        device = device 
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False).pooler_output

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )

        if pooled_prompt_embeds is None:
            if prompt_2 is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
if __name__ == "__main__":
    import os
    import torch
    from diffusers.utils import export_to_video
    model = HunyuanV2V(None)
    model.bfloat16()
    model.to(5)