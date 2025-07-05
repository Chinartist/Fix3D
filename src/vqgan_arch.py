'''
VQGAN code, adapted from the original created by the Unleashing Transformers authors:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


from src.base import normalize, swish, Downsample, Upsample, ResBlock


#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.emb_dim**-0.5)
        # self.simvq = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.simvq = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=False),
        )
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3,4, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        quant_codebook  = self.simvq(self.embedding.weight)
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (quant_codebook**2).sum(1) - \
            2 * torch.matmul(z_flattened, quant_codebook.t())

        mean_distance = torch.mean(d)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim=1, largest=False)
        # [0-1], higher score, higher confidence
        # min_encoding_scores = torch.exp(-min_encoding_scores/10)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, quant_codebook).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 4, 1, 2,3).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance,
            "codebook_size": self.codebook_size,
            "indices_gt": min_encoding_indices.squeeze(1)
            }
    @torch.no_grad()
    def get_indices_gt(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3,4, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        quant_codebook  = self.simvq(self.embedding.weight)
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (quant_codebook**2).sum(1) - \
            2 * torch.matmul(z_flattened, quant_codebook.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices,{
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": torch.mean(d),
            "codebook_size": self.codebook_size,
        }
    def get_codebook_feat(self, indices):
        indices = indices.permute(0, 2, 3,4, 1).contiguous()
        shape = indices.shape
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        quant_codebook  = self.simvq(self.embedding.weight)
        z_q = torch.matmul(min_encodings.to(quant_codebook),  quant_codebook)
        z_q = z_q.reshape(shape[0], shape[1], shape[2],shape[3], -1).permute(0, 4, 1, 2,3).contiguous()
        return z_q



from copy import deepcopy

class Unet(nn.Module):
    def __init__(self,input_dim=128,output_dim=10240,num_layer=2,num_mid_layer=2):
        super().__init__()

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(0,num_layer):
            encoder = []
            encoder+=[ResBlock(input_dim*2**i, input_dim*2**(i+1))]
            encoder+= [ResBlock(input_dim*2**(i+1), input_dim*2**(i+1)) for _ in range(num_mid_layer)]
            encoder.append(nn.Sequential(normalize(input_dim*2**(i+1)),nn.Conv2d(input_dim*2**(i+1), input_dim*2**(i+1), kernel_size=3, stride=1, padding=1),))
            encoder.append(Downsample(input_dim*2**(i+1)))
            self.encoder.append(nn.Sequential(*encoder))
        # self.local = LlamaTransformer
        skip_con = []
        for i in range(0,num_layer):
            decoder = []
            decoder.append(Upsample(input_dim*2**(num_layer-i)))
            decoder+=[ResBlock(input_dim*2**(num_layer-i), input_dim*2**(num_layer-i)) for _ in range(num_mid_layer)]
            decoder.append(ResBlock(input_dim*2**(num_layer-i), input_dim*2**(num_layer-i-1)))
            decoder.append(nn.Sequential(normalize(input_dim*2**(num_layer-i-1)),
                                   nn.Conv2d(input_dim*2**(num_layer-i-1), input_dim*2**(num_layer-i-1), kernel_size=3, stride=1, padding=1),))
            skip_con.append(nn.Conv2d(input_dim*2**(num_layer-i-1), input_dim*2**(num_layer-i-1), kernel_size=3, stride=1, padding=1))
            self.decoder.append(nn.Sequential(*decoder))
        conv_out = [ResBlock(input_dim, input_dim) for _ in range(num_mid_layer)]
        conv_out.append(nn.Sequential(normalize(input_dim),nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)))
        self.conv_out = nn.Sequential(*conv_out)
        self.skip_con = nn.ModuleList(skip_con)
        self.num_layer = num_layer
        self.num_mid_layer = num_mid_layer
    def forward(self, x):
        skip = []
        for i in range(0,self.num_layer):
            skip.append(x)
            x = self.encoder[i](x)
        for i in range(0,self.num_layer):
            x = self.decoder[i](x)
            x = x + self.skip_con[i](skip[self.num_layer-i-1])
        x = self.conv_out(x)
        return x



class LatentCodeFormer(nn.Module):
    def __init__(self, input_dim=4,hidden_dim=128,output_dim=4,num_layer=6,codebook_size=10240,num_mid_layer=2):
        super().__init__()
        self.codebook_size = codebook_size
        hq_encoder = [nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            normalize(hidden_dim),
        )]
        for i in range(num_layer):
            hq_encoder.append(ResBlock(hidden_dim, hidden_dim))
        hq_encoder.append(nn.Sequential(normalize(hidden_dim),nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)))
        self.hq_encoder = nn.Sequential(*hq_encoder)
        
        lq_encoder = {"encoder":deepcopy(nn.Sequential(*hq_encoder)),"classifier":Unet(input_dim=hidden_dim,output_dim=codebook_size,num_layer=2,num_mid_layer=num_mid_layer)}
        self.lq_encoder = nn.ModuleDict(lq_encoder)
        self.beta = 0.25 #0.25
        self.quantize = VectorQuantizer(self.codebook_size,hidden_dim, self.beta,)
        decoder = []
        for i in range(num_layer):
            decoder.append(ResBlock(hidden_dim, hidden_dim))
        self.decoder = nn.Sequential(*decoder)
        self.conv_out = nn.Sequential(normalize(hidden_dim),nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1))
    def train_stage(self, stage):
        if stage==1:
            self.requires_grad_(True)
            self.quantize.embedding.requires_grad_(False)
            self.lq_encoder.requires_grad_(False)
        elif stage==3:
            self.requires_grad_(False)
            self.lq_encoder.requires_grad_(True)
    def hq_forward(self, x):
        enc = self.hq_encoder(x)
        quant_enc, codebook_loss, quant_stats = self.quantize(enc)
        dec = self.decoder(quant_enc)
        out = self.conv_out(dec)
        return out, codebook_loss,{0:quant_stats}
    def lq_forward(self, x,gt):
        enc = self.lq_encoder["encoder"](x)
        quant_logits = self.lq_encoder["classifier"](enc)
        with torch.no_grad():
            quant_indices = quant_logits.argmax(dim=1, keepdim=True)
            quant_enc = self.quantize.get_codebook_feat(quant_indices)
            dec = self.decoder(quant_enc)
            out = self.conv_out(dec)
        loss = 0
        with torch.no_grad():
          indices_gt,quant_stats = self.quantize.get_indices_gt(self.hq_encoder(gt))
        loss = F.cross_entropy(quant_logits.permute(0,2,3,1).reshape(-1,quant_logits.shape[1]), indices_gt)
        return out,loss,{0:quant_stats}
    
    
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x):
        return self.main(x)
from transformers import SamModel
class SamPercepLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        self.model = model.vision_encoder.cuda().requires_grad_(False)
        self.mean = [0.485,
    0.456,
    0.406]
        self.std = [
    0.229,
    0.224,
    0.225
  ]
        self.loss_fn = torch.nn.MSELoss()

    def encode(self, image):
        image = image*0.5+0.5
        image = torch.nn.functional.interpolate(image, size=self.model.patch_embed.image_size, mode="bilinear")
        mean = self.mean
        std = self.std
        mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(image.device)
        std = torch.tensor(std).reshape(1, 3, 1, 1).to(image.device)
        image = (image - mean) / std
        hidden_states = self.model(image,output_hidden_states=True).hidden_states
        hidden_states = torch.cat(hidden_states)
        return hidden_states
    def forward(self,pred, target):
        pred_hidden_states = self.encode(pred)
        target_hidden_states = self.encode(target)
        loss = 0.
        for i in range(len(pred_hidden_states)):
            loss += self.loss_fn(pred_hidden_states[i], target_hidden_states[i])
        return loss
        
        
from transformers import AutoModel
class DinoPercepLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.model = model.cuda().requires_grad_(False)
        self.mean = [0.485,
    0.456,
    0.406]
        self.std = [
    0.229,
    0.224,
    0.225
  ]
        self.loss_fn = torch.nn.MSELoss()
        self.image_size = self.model.embeddings.patch_embeddings.image_size
    def encode(self, image):  
        image = image*0.5+0.5
        image = torch.nn.functional.interpolate(image, size=self.image_size, mode="bilinear")
        mean = self.mean
        std = self.std
        mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(image.device)
        std = torch.tensor(std).reshape(1, 3, 1, 1).to(image.device)
        image = (image - mean) / std
        hidden_states = self.model(pixel_values=image,output_hidden_states=True).hidden_states
        hidden_states = torch.cat(hidden_states)
        return hidden_states
    def forward(self,pred, target):
        pred_hidden_states = self.encode(pred)
        target_hidden_states = self.encode(target)
        loss = 0.
        for i in range(len(pred_hidden_states)):
            loss += self.loss_fn(pred_hidden_states[i], target_hidden_states[i])
        return loss
    
from src.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
#depthanything perc
class DepthAnythingv2PercepLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        encoder = 'vitb'
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(torch.load(f'/nvme0/public_data/Occupancy/proj/cache/depth_anything_v2/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = depth_anything.to("cuda").eval().requires_grad_(False)
        self.mean = [0.485,
    0.456,
    0.406]
        self.std = [
    0.229,
    0.224,
    0.225
  ]
        self.loss_fn = torch.nn.MSELoss()
        self.image_size = self.model.pretrained.patch_embed.img_size
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    def encode(self, image):
        image = image*0.5+0.5
        image = torch.nn.functional.interpolate(image, size=self.image_size, mode="bilinear")
        mean = self.mean
        std = self.std
        mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(image.device)
        std = torch.tensor(std).reshape(1, 3, 1, 1).to(image.device)
        image = (image - mean) / std
        depth,hidden_states = self.model(image)
        hidden_states0 = [hidden_states[i][0] for i in range(len(hidden_states))]
        hidden_states1 = [hidden_states[i][1] for i in range(len(hidden_states))]
        hidden_states = hidden_states0 + hidden_states1
        return depth,hidden_states
    def forward(self,pred, target):
        pred_depth, pred_hidden_states = self.encode(pred)
        target_depth, target_hidden_states = self.encode(target)
        loss = 0.
        # for i in range(len(pred_hidden_states)):
        #     loss += self.loss_fn(pred_hidden_states[i], target_hidden_states[i])
        loss += self.loss_fn(pred_depth, target_depth)
        pred_depth =self.depth_post_batch(pred_depth)
        target_depth = self.depth_post_batch(target_depth)
        return loss, pred_depth, target_depth
    def depth_post(self,depth):
        depth = depth.float().detach().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = (self.cmap(depth)[:, :, :3])[:, :, ::-1]
        return depth
    def depth_post_batch(self,depth):
        outs = []
        for i in range(depth.shape[0]):
            outs.append(self.depth_post(depth[i]))
        outs =np.stack(outs, axis=0)
        return outs
#dino discriminator
from torchvision.transforms import RandomCrop, Normalize
from src.diffaug import DiffAugment
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, for_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            for_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return for_real
        target_val = (self.real_label_val if for_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, for_real, for_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            for_real (bool): Whether the targe is real or fake.
            for_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        if self.gan_type == 'hinge':
            if for_disc:  # for discriminators in hinge-gan
                input = -input if for_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            target_label = self.get_target_label(input, for_real)
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if for_disc else loss * self.loss_weight

class DiscHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in =ResBlock(in_channels, in_channels)
        self.conv_out = nn.Sequential(
            normalize(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),   
        )
    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_out(x)
        return x
from transformers import Dinov2Model
class DinoDiscriminator(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        model:Dinov2Model = Dinov2Model.from_pretrained("/nvme0/public_data/Occupancy/proj/cache/facebook/dinov2-small")
        from safetensors import safe_open

        tensors = {}
        with safe_open("/nvme0/public_data/Occupancy/proj/cache/facebook/dinov2-small/model.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        self.model = Dinov2Model(model.config)
        self.model.load_state_dict(tensors, strict=True)
        self.model.requires_grad_(False)
        self.hooks = [2,5,8,11]
        self.image_size = self.model.embeddings.patch_embeddings.image_size
        self.num_patches = int(self.model.embeddings.patch_embeddings.num_patches**0.5)
        self.diffaug = False
        self.p_crop = 0.3
        self.mean = [0.485,
    0.456,
    0.406]
        self.std = [
    0.229,
    0.224,
    0.225
  ]     
        self.heads = nn.ModuleList()
        for i in range(len(self.hooks)):
            self.heads.append(DiscHead(self.model.config.hidden_size, 1))
        self.heads.requires_grad_(True)
        self.loss_fn = GANLoss('hinge')
    def set_g(self,):
        self.requires_grad_(False)
    def set_d(self,):
        self.model.requires_grad_(False)
        self.heads.requires_grad_(True)
    def save_model(self, path):
        sd = {}
        sd["heads"] = self.heads.state_dict()
        torch.save(sd, path)
    def load_model(self, path):
        print(f"loading disc model from {path}")
        sd = torch.load(path)
        self.heads.load_state_dict(sd["heads"])
    def encode(self,x):
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')
        # Transform to [0, 1].
        x = x * 0.5 + 0.5
        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.image_size[0] and np.random.random() < self.p_crop:
            x = RandomCrop(self.image_size)(x)
        # Resize to the input size of the model.
        x = torch.nn.functional.interpolate(x, size=self.image_size, mode="bilinear")
        # Normalize.
        mean = self.mean
        std = self.std
        mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(x.device)
        std = torch.tensor(std).reshape(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        # Forward pass.
        hidden_states = self.model(x,output_hidden_states=True).hidden_states[1:]
        # Get the outputs of the heads.
        outs = []
        for i, hook in enumerate(self.hooks):
            h = hidden_states[hook][:,1:]
            h = h.reshape(h.shape[0],  self.num_patches, self.num_patches,h.shape[2])
            h = h.permute(0,3,1,2).contiguous()
            h = self.heads[i](h)
            h = h.reshape(h.shape[0],-1)
            outs.append(h)
        outs = torch.cat(outs, dim=1)
        return outs
    def forward(self, x,for_real=False, for_disc=False):
        if for_disc:
            self.set_d()
        else:
            self.set_g()
        logdits = self.encode(x)
        loss = self.loss_fn(logdits,for_real, for_disc)
        return loss
if __name__ == "__main__":
    # test the LatentCodeFormer
    # model = LatentCodeFormer(input_dim=4,hidden_dim=128,output_dim=4,num_layer=6,codebook_size=10240)
    # x = torch.randn(1, 4, 64, 64)
    # out = model.lq_forward(x)
    # model = Unet(input_dim=128,output_dim=128,num_layer=4,num_mid_layer=2)
    model = DinoDiscriminator()
    x = torch.randn(2, 3, 64, 64).cuda()
    out = model(x)
    print(out)

    