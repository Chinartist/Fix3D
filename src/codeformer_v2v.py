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
from vqgan_arch import GANLoss,DiscHead,VectorQuantizer
from diffusers.models.transformers.transformer_wan import FP32LayerNorm, Attention, FeedForward, WanAttnProcessor2_0, WanRotaryPosEmbed,ModelMixin,\
         ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin,register_to_config,logger,USE_PEFT_BACKEND,scale_lora_layers,\
         Transformer2DModelOutput,unscale_lora_layers,math
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        # self.attn2 = Attention(
        #     query_dim=dim,
        #     heads=num_heads,
        #     kv_heads=num_heads,
        #     dim_head=dim // num_heads,
        #     qk_norm=qk_norm,
        #     eps=eps,
        #     bias=True,
        #     cross_attention_dim=None,
        #     out_bias=True,
        #     added_kv_proj_dim=added_kv_proj_dim,
        #     added_proj_bias=True,
        #     processor=WanAttnProcessor2_0(),
        # )
        # self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table #+ temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        # norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        # attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        # hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

class TransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        # self.condition_embedder = WanTimeTextImageEmbedding(
        #     dim=inner_dim,
        #     time_freq_dim=freq_dim,
        #     time_proj_dim=inner_dim * 6,
        #     text_embed_dim=text_dim,
        #     image_embed_dim=image_dim,
        #     pos_embed_seq_len=pos_embed_seq_len,
        # )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor=None,
        encoder_hidden_states: torch.Tensor=None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        #     timestep, encoder_hidden_states, encoder_hidden_states_image
        # )
        # timestep_proj = timestep_proj.unflatten(1, (6, -1))
        temb=None
        timestep_proj=None
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table \
                        #+ temb.unsqueeze(1)
                        ).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)




class CodeformerV2V(nn.Module):
    def __init__(self,):
        super().__init__()
        model_id = "/nvme0/public_data/Occupancy/proj/cache/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae:AutoencoderKLWan = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)

        codebook_size = 2**15
        wantransformer:WanTransformer3DModel = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        transformer = TransformerModel.from_config(wantransformer.config,out_channels=codebook_size,torch_dtype=torch.bfloat16)
        sd=wantransformer.state_dict()
        sd = {k:v for k,v in sd.items() if "proj_out" not in k}
        transformer.load_state_dict(sd,strict=False)
        self.quantizer = VectorQuantizer(codebook_size=codebook_size,emb_dim=vae.z_dim,beta=0.25).bfloat16()
        self.transformer = transformer
        self.vae = vae
        self.requires_grad_(False)
        self.bfloat16()
    def set_train(self,stage,pretrained=None):
        self.stage = stage
        if stage==0:
            if pretrained is not None:
                print(f"Loading pretrained weights from {pretrained}")
                sd = torch.load(pretrained, map_location="cpu")
                self.vae.load_state_dict(sd["vae"])
                self.quantizer.load_state_dict(sd["quantizer"])
            self.requires_grad_(False)
            for name, param in self.vae.named_parameters():
                # if "up" not in name and "down" not in name:
                    param.requires_grad_(True)
            self.quantizer.simvq.requires_grad_(True)
        elif stage==1:
            self.requires_grad_(False)
            self.transformer.requires_grad_(True)
    def forward(self,batch,):
        x_src = batch["x_src"][:,:,None]
        x_tgt = batch["x_tgt"][:,:,None]
        dtype = self.transformer.dtype
        device = self.transformer.device
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                device, dtype
            )
        if self.stage==0:
            latents = (self.vae.encode(x_tgt).latent_dist.sample()- latents_mean)* latents_std
            quant_latents, quant_loss,quant_stats = self.quantizer(latents)
            image_pred = self.vae.decode((quant_latents /latents_std)+latents_mean,return_dict=False)[0]
            image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
            loss = {}
            loss["quant_loss"] = quant_loss
            return image_pred,loss,quant_stats
        
        elif self.stage==1:
            cond = (self.vae.encode(x_src).latent_dist.sample()- latents_mean)* latents_std
            sample = (self.vae.encode(x_tgt).latent_dist.sample()- latents_mean)* latents_std
            quant_logits = self.transformer(
                hidden_states=cond,return_dict=False,
        )[0]
            with torch.no_grad():
                indices_pred = quant_logits.argmax(dim=1, keepdim=True)
                quant_latents= self.quantizer.get_codebook_feat(indices_pred)
                indices_gt,quant_stats = self.quantizer.get_indices_gt(sample)
            loss = {}
            loss["ce_loss"] = F.cross_entropy(quant_logits.permute(0,2,3,4,1).reshape(-1,quant_logits.shape[1]), indices_gt)
            quant_latents = (quant_latents / latents_std) + latents_mean
            image_pred = self.vae.decode(quant_latents,return_dict=False)[0]
            image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
            return image_pred,loss,quant_stats
        
        elif self.stage==None:
            with torch.no_grad():
                cond = (self.vae.encode(x_src).latent_dist.sample()- latents_mean)* latents_std
            sample = (self.vae.encode(x_tgt).latent_dist.sample()- latents_mean)* latents_std
            quant_latents_gt, quant_loss,quant_stats = self.quantizer(sample)
            indices_gt = quant_stats["indices_gt"].detach()

            quant_logits = self.transformer(
                hidden_states=cond,return_dict=False,
            )[0]
            with torch.no_grad():
                indices_pred = quant_logits.argmax(dim=1, keepdim=True)
                quant_latents_pred= self.quantizer.get_codebook_feat(indices_pred)


            image_pred_gt = self.vae.decode((quant_latents_gt /latents_std)+latents_mean,return_dict=False)[0]
            image_pred_gt  = image_pred_gt[:,:,0].clamp(-1.0, 1.0)
            with torch.no_grad():
                image_pred = self.vae.decode((quant_latents_pred /latents_std)+latents_mean,return_dict=False)[0]
                image_pred  = image_pred[:,:,0].clamp(-1.0, 1.0)
            loss = {}
            loss["quant_loss"] = quant_loss
            loss["ce_loss"] = F.cross_entropy(quant_logits.permute(0,2,3,4,1).reshape(-1,quant_logits.shape[1]), indices_gt)
            return image_pred,image_pred_gt,loss,quant_stats
    def save_model(self,path):
        if self.stage==0:
            sd = {}
            sd["vae"] = self.vae.state_dict()
            sd["quantizer"] = self.quantizer.state_dict()
            torch.save(sd, path.replace("RP","vqvae"))
        elif self.stage==1:
            sd = {}
            sd["transformer"] = self.transformer.state_dict()
            torch.save(sd, path.replace("RP","transformer"))

        

        
        
if __name__ == "__main__":
    model = CodeformerV2V()
    model.eval()
    x_src = torch.randn(1, 3, 64, 64).to(torch.bfloat16)
    x_tgt = torch.randn(1,3, 64, 64).to(torch.bfloat16)
    batch = {"x_src": x_src, "x_tgt": x_tgt}
    model(batch, mode=0)
    model(batch,mode=1)
    model(batch,mode=2)