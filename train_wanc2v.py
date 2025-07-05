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

import random
import wandb

from src.vqgan_arch import DinoPercepLoss,DepthAnythingv2PercepLoss,SamPercepLoss

from src.wan_c2v import WanC2V
from src.my_utils.training_utils import parse_args_paired_training,Canny2ImageDataset
from rich import print
from accelerate import DistributedDataParallelKwargs,FullyShardedDataParallelPlugin,DeepSpeedPlugin
from rich.progress import Progress,track, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn,SpinnerColumn ,RenderableColumn,MofNCompleteColumn
import warnings
from accelerate.utils import get_active_deepspeed_plugin
warnings.filterwarnings("ignore", category=UserWarning)
from src.loss_utils import ssim,ms_ssim
import torch.nn as nn
import tensorboard
from torch.utils.tensorboard import SummaryWriter
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

    dataset_train = Canny2ImageDataset(dataset_folder=args.dataset_folder,  split="train",)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)

    net_v2v = WanC2V(pretrained_path=args.pretrained_path,)
    net_v2v.set_train()
    # if args.gradient_checkpointing:
    #     net_v2v.unet.enable_gradient_checkpointing()
        # net_reg.unet_fix.enable_gradient_checkpointing()
        # net_reg.unet_update.enable_gradient_checkpointing()
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
    # start the training loop
    global_step = 0 
    for epoch in range(0, args.num_training_epochs):
        pbar.reset(train_task,)
        for step, batch in enumerate(dl_train):
            l_acc = [net_v2v]
            with accelerator.accumulate(*l_acc):

                B = batch["x_src"].shape[0]
                batch["prompt"] = ["" for _ in range(B)]
              
                # forward pass
                batch["x_src"] = batch["x_src"].cuda().to(weight_dtype)
                batch["x_cond"] = batch["x_cond"].cuda().to(weight_dtype)
                x_tgt_pred,extra_loss= net_v2v(batch)
                x_tgt_pred = x_tgt_pred*0.5 + 0.5
                x_src = batch["x_cond"][:,:,0]*0.5+0.5
                x_tgt = batch["x_src"][:,:,0]*0.5+0.5

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
                    pbar.update(global_task, advance=1)
                    pbar.update(train_task, advance=1,description=f"[red]Epoch: {epoch:04d}")
                    # viz some images
                    concate_image = torch.cat([x_src,x_tgt_pred,x_tgt], dim=-1)
                    if global_step % args.viz_freq == 1:  
                        log_dict = {
                 
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/diff_pred": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            # "train/gt_depth": [wandb.Image(gt_depth[idx], caption=f"idx={idx}") for idx in range(B)],
                            # "train/pred_depth": [wandb.Image(pred_depth[idx], caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 10:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_v2v).save_model(outf)
                 
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
