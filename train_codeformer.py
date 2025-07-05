import os
import gc
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler


import wandb

from vqgan_arch import DinoPercepLoss,DepthAnythingv2PercepLoss,SamPercepLoss,DinoDiscriminator

from codeformer_v2v import CodeformerV2V
from my_utils.training_utils import parse_args_paired_training,RenderSRDataset,RenderDataset
from rich import print
from accelerate import DistributedDataParallelKwargs,FullyShardedDataParallelPlugin,DeepSpeedPlugin
from rich.progress import Progress,track, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn,SpinnerColumn ,RenderableColumn,MofNCompleteColumn
import warnings
from accelerate.utils import get_active_deepspeed_plugin
warnings.filterwarnings("ignore", category=UserWarning)
from loss_utils import ssim
import torch.nn as nn
import tensorboard
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

class codebook_utils():
    def __init__(self,):
        self.codebook_used = {}
    def update(self, x):
        for i in range(len(x)):
            if i not in self.codebook_used:
                self.codebook_used[i] = torch.zeros(x[i]["codebook_size"])
            self.codebook_used[i][x[i]["min_encoding_indices"].detach().cpu().reshape(-1)]+=1
    def get_utils(self):
        return {k:(self.codebook_used[k]>100).sum()/len(self.codebook_used[k]) for k in self.codebook_used}
def main(args):
    
    # fsdp_plugin = FullyShardedDataParallelPlugin(fsdp_version=2)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False,)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # kwargs_handlers = [ddp_kwargs],
        deepspeed_plugins=DeepSpeedPlugin(zero_stage=2),
        # fsdp_plugin=fsdp_plugin,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    dataset_train = RenderSRDataset(dataset_folder=args.dataset_folder,  split="train",)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = RenderSRDataset(dataset_folder=args.dataset_folder, split="val")
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=args.dataloader_num_workers)

    net_v2v = CodeformerV2V()
    net_v2v.set_train(args.train_stage,"/nvme0/public_data/Occupancy/proj/img2img-turbo/outputs/pix2pix_turbo/stage0_render/checkpoints/vqvae_18710.pkl")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # make the optimizer
    
    layers_to_opt = [p for n, p in net_v2v.named_parameters() if p.requires_grad] 
    with open(os.path.join(args.output_dir, f"trainable_layers.txt"), "w") as f:
        for n, _p in net_v2v.named_parameters():
            if _p.requires_grad:
                f.write(f"{n} {_p.shape}\n")
        
    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    max_train_steps = len(dl_train) // args.gradient_accumulation_steps * args.num_training_epochs
    lr_warmup_steps = int(args.lr_warmup_rate * max_train_steps)
    if accelerator.is_main_process:
        print(len(layers_to_opt), " layers to optimize ")
        print(f"Total training steps: {max_train_steps}, warmup steps: {lr_warmup_steps}")
        print("Total training steps of each epoch: ", len(dl_train)) 
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    # Prepare everything with our `accelerator`.

    net_v2v, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_v2v, optimizer,dl_train, lr_scheduler
    )
  
    #other model not trainable
    net_lpips = lpips.LPIPS(net='vgg').cuda()

    net_depth = DepthAnythingv2PercepLoss()
    net_lpips.requires_grad_(False)
    net_depth.requires_grad_(False)
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_vlm = ram(pretrained="/nvme0/public_data/Occupancy/proj/SeeSR/preset/models/ram_swin_large_14m.pth",
            pretrained_condition="",
            image_size=384,
            vit='swin_l')
    model_vlm.eval()
    model_vlm.to("cuda", dtype=torch.float16)

    #get the dtype of the weights
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if accelerator.is_main_process:
        print(f"Using weight dtype: {weight_dtype}")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    pbar = Progress(TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    MofNCompleteColumn(),
    SpinnerColumn(),
    RenderableColumn(),
    disable=not accelerator.is_main_process)
    pbar.start()
    global_task = pbar.add_task("[green]Global steps", total=len(dl_train) * args.num_training_epochs)
    train_task = pbar.add_task("[red]Training...", total=len(dl_train))
    val_task = pbar.add_task("[blue]Validation...", total=args.num_samples_eval)
    # start the training loop
    global_step = 0
    CU = codebook_utils()
    lossG = torch.tensor(0).cuda()
    lossD_real = torch.tensor(0).cuda()
    lossD_fake = torch.tensor(0).cuda()
    tb_writer = SummaryWriter(f"/nvme0/public_data/Occupancy/proj/img2img-turbo/outputs/pix2pix_turbo/stage0_render/log_codeformer_stage{args.train_stage}_nusc3rc")
    for epoch in range(0, args.num_training_epochs):
        pbar.reset(train_task,)
        for step, batch in enumerate(dl_train):
            l_acc = [net_v2v]
            with accelerator.accumulate(*l_acc):

                x_src = batch["x_src"]
                x_tgt = batch["x_tgt"]
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                batch["prompt"] = [f'{each_caption}' for each_caption in caption]
                batch["negative_prompt"] = [args.neg_prompt for _ in range(len(batch["prompt"]))]
                B, C, H, W = x_src.shape
                # forward pass
                batch["x_src"] = batch["x_src"].cuda().to(weight_dtype)
                batch["x_tgt"] = batch["x_tgt"].cuda().to(weight_dtype)
             
                x_tgt_pred,extra_loss,quant_stats= net_v2v(batch)
                quant_stats = {0:quant_stats}
                if quant_stats is not None:
                    CU.update(quant_stats)
           
                # # Reconstruction loss
               
                loss_rec = F.mse_loss(x_tgt_pred,batch["x_tgt"], reduction="mean") * args.lambda_rec
                # loss_ssim = (1 - ssim(x_tgt_pred*0.5+0.5, batch["x_tgt"]*0.5+0.5))* args.lambda_ssim
                loss_lpips = net_lpips(x_tgt_pred, batch["x_tgt"]).mean() * args.lambda_lpips
                loss_depth,pred_depth,gt_depth = net_depth(x_tgt_pred, batch["x_tgt"]) 
                loss_depth = loss_depth * args.lambda_depth
                extra_loss["loss_rec"] = loss_rec
                # extra_loss["loss_ssim"] = loss_ssim
                extra_loss["loss_lpips"] = loss_lpips
                extra_loss["loss_depth"] = loss_depth   
                loss = 0.0
                for k,v in extra_loss.items():
                    loss += v
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
      
    
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                
                global_step += 1
                # print((accelerator.gather(torch.tensor(global_step).unsqueeze(0).cuda()).sum().item()))
                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    for k, v in extra_loss.items():
                        logs[k] = v.detach().item()
                        tb_writer.add_scalar(f"train/{k}", v.detach().item(), global_step)
                    if quant_stats is not None:
                        codebook_used_info = CU.get_utils()
                        for k in codebook_used_info:
                            logs[f"codebook_used_{k}"] = codebook_used_info[k]
                            tb_writer.add_scalar(f"train/codebook_used_{k}", codebook_used_info[k], global_step)
                    pbar.update(global_task, advance=1)
                    pbar.update(train_task, advance=1,description=f"[red]Epoch: {epoch:04d}")
                    # viz some images
                    concate_image = torch.cat([x_src,x_tgt_pred,x_tgt], dim=-1)
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/concate_image": [wandb.Image(concate_image[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)]
                            # "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            # "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            # "train/diff_pred": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            # "train/gt_depth": [wandb.Image(gt_depth[idx], caption=f"idx={idx}") for idx in range(B)],
                            # "train/pred_depth": [wandb.Image(pred_depth[idx], caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 10:
                        outf = os.path.join(args.output_dir, "checkpoints", f"RP_{global_step}.pkl")
                        accelerator.unwrap_model(net_v2v).save_model(outf)
                    # compute validation set FID, rec, LPIPS
                    #save prompt
                    # eval_prompt = []
                    # if global_step % args.eval_freq == 1:
                    #     l_rec, l_lpips = [], []
                    #     for step, batch_val in enumerate(dl_val):
                    #         if step >= args.num_samples_eval:
                    #             break
                    #         x_src = batch_val["x_src"].cuda()
                    #         x_tgt = batch_val["x_tgt"].cuda()
                    #         x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                    #         caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                    #         batch_val["prompt"] = [f'{each_caption}' for each_caption in caption]
                    #         eval_prompt+= batch_val["prompt"]                      
                    #         B, C, H, W = x_src.shape
                    #         assert B == 1, "Use batch size 1 for eval."
                    #         batch_val["x_src"] = batch_val["x_src"].cuda().to(weight_dtype)
                    #         batch_val["x_tgt"] = batch_val["x_tgt"].cuda().to(weight_dtype)
                    #         with torch.no_grad():
                    #             # forward pass
                    #             x_tgt_pred= accelerator.unwrap_model(net_v2v).infer(batch_val)
                    #             # compute the reconstruction losses
                    #             loss_rec = F.l1_loss(x_tgt_pred, x_tgt, reduction="mean")
                    #             loss_lpips = net_lpips(x_tgt_pred, x_tgt.float()).mean()
          
                    #             l_rec.append(loss_rec.item())
                    #             l_lpips.append(loss_lpips.item())
                                
                    #             concate_img = torch.cat([(x_src* 0.5 + 0.5).clamp(0, 1), (x_tgt * 0.5 + 0.5).clamp(0, 1),(x_tgt_pred * 0.5 + 0.5).clamp(0, 1) ], dim=-2)
                    #             concate_img = transforms.ToPILImage()(concate_img[0].cpu())
                    #             os.makedirs(os.path.join(args.output_dir, "eval", "val_imgs"), exist_ok=True)
                    #             outf = os.path.join(args.output_dir, "eval","val_imgs",f"val_{step}.png")
                    #             concate_img.save(outf)
                    #             pbar.update(val_task, advance=1)
                    #     with open(os.path.join(args.output_dir, "eval", f"prompt.txt"), "w") as f:
                    #         for i in range(len(eval_prompt)):
                    #             f.write(f"{i}: {eval_prompt[i]}\n")
                        
                    #     pbar.reset(val_task)
                    #     logs["val/rec"] = np.mean(l_rec)
                    #     logs["val/lpips"] = np.mean(l_lpips)
                 
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
