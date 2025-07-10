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

from src.vqgan_arch import DinoPercepLoss,DepthAnythingv2PercepLoss,SamPercepLoss,DinoDiscriminator
from src.ram.models.ram_lora import ram
from src.ram import inference_ram as inference
from src.sd2_i2i import SD2I2I,SD2Reg
from src.my_utils.training_utils import parse_args_paired_training,RenderSRDataset,RenderDataset
from rich import print
from accelerate import DistributedDataParallelKwargs,FullyShardedDataParallelPlugin,DeepSpeedPlugin
from rich.progress import Progress,track, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn,SpinnerColumn ,RenderableColumn,MofNCompleteColumn
import warnings
from accelerate.utils import get_active_deepspeed_plugin
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn
import tensorboard
from torch.utils.tensorboard import SummaryWriter

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
    accelerator1 = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers = [ddp_kwargs],
    )
    accelerator2= Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers = [ddp_kwargs],
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

    dataset_train = RenderDataset(dataset_folder=args.dataset_folder,  split="train",)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    net_v2v = SD2I2I(pretrained_path=args.pretrained_path)
    net_v2v.set_train()
    if args.gradient_checkpointing:
        net_v2v.unet.enable_gradient_checkpointing()
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
    net_reg = SD2Reg()
    net_reg.set_train()
    net_disc = DinoDiscriminator()
    net_disc.train()
    with open(os.path.join(args.output_dir, f"trainable_layers_disc.txt"), "w") as f:
        for n, _p in net_disc.named_parameters():
            if _p.requires_grad:
                f.write(f"{n} {_p.shape}\n")
    with open(os.path.join(args.output_dir, f"trainable_layers_reg.txt"), "w") as f:
        for n, _p in net_reg.named_parameters():
            if _p.requires_grad:
                f.write(f"{n} {_p.shape}\n")
    layers_to_opt_reg = []
    for n, _p in net_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    layers_to_opt_disc = [p for n, p in net_disc.named_parameters() if p.requires_grad]
    optimizer_disc = torch.optim.AdamW(layers_to_opt_disc, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    net_disc ,optimizer_disc= accelerator1.prepare(net_disc,optimizer_disc)
    net_reg,optimizer_reg = accelerator2.prepare(net_reg,optimizer_reg)
    #other model not trainable
    net_lpips = lpips.LPIPS(net='vgg').cuda()

    net_depth = DepthAnythingv2PercepLoss()
    net_lpips.requires_grad_(False)
    net_depth.requires_grad_(False)
    
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
    # start the training loop
    global_step = 0
    lossG = torch.tensor(0).cuda()
    lossD_real = torch.tensor(0).cuda()
    lossD_fake = torch.tensor(0).cuda()
   
    for epoch in range(0, args.num_training_epochs):
        pbar.reset(train_task,)
        for step, batch in enumerate(dl_train):
            l_acc = [net_v2v]
            with accelerator.accumulate(*l_acc):

                x_src = batch["x_src"]
                x_tgt = batch["x_tgt"]
                B, C, H, W = x_src.shape
                batch["prompt"] = [f'' for _ in range(B)]
                # forward pass
                batch["x_src"] = batch["x_src"].cuda().to(weight_dtype)
                batch["x_tgt"] = batch["x_tgt"].cuda().to(weight_dtype)
                mask_rec = batch["mask_rec"].cuda().to(weight_dtype)
                x_tgt_pred,latents_pred,latents_gt,extra_loss= net_v2v(batch)
                
                # # Reconstruction loss
                if mask_rec.sum()>0:
                    loss_rec = F.mse_loss(x_tgt_pred[mask_rec>0],batch["x_tgt"][mask_rec>0], reduction="mean") * args.lambda_rec
              
                    loss_lpips = net_lpips(x_tgt_pred[mask_rec>0], batch["x_tgt"][mask_rec>0]).mean() * args.lambda_lpips
                    loss_depth,pred_depth,gt_depth = net_depth(x_tgt_pred[mask_rec>0], batch["x_tgt"][mask_rec>0]) 
                    loss_depth = loss_depth * args.lambda_depth
                    extra_loss["loss_rec"] = loss_rec
                    extra_loss["loss_lpips"] = loss_lpips
                    extra_loss["loss_depth"] = loss_depth   
                
                if args.lambda_vsd>0:
                    caption = inference(ram_transforms(x_tgt*0.5+0.5).to(dtype=torch.float16), model_vlm)
                    batch["prompt"] = [f'{each_caption}' for each_caption in caption]
                    batch["negative_prompt"] = [f'{args.neg_prompt}' for _ in range(B)]
                    if torch.cuda.device_count() > 1:
                        prompt_embeds = net_reg.module.encode_prompt(batch["prompt"])
                        neg_prompt_embeds = net_reg.module.encode_prompt(batch["negative_prompt"])
                        loss_kl = net_reg.module.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, ) * args.lambda_vsd
                    else:
                        prompt_embeds = net_reg.encode_prompt(batch["prompt"])
                        neg_prompt_embeds = net_reg.encode_prompt(batch["negative_prompt"])
                        loss_kl = net_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, ) * args.lambda_vsd
                    extra_loss["loss_kl"] = loss_kl
                
                """
                Generator loss: fool the discriminator
                """ 
                if global_step  > args.disc_start_iter and args.disc_start_iter>=0:
                    if global_step == args.disc_start_iter+1:
                        print("Generator loss starts at global step: ", global_step)
                    lossG = net_disc(x=x_tgt_pred,for_real=True, for_disc=False) * args.lambda_gan
                    extra_loss["lossG"] = lossG
                loss = 0.0
                for k,v in extra_loss.items():
                    loss += v
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                """
                diff loss: let lora model closed to generator 
                """
                if args.lambda_vsd>0:
                    for _ in range(args.reg_freq) :
                        if torch.cuda.device_count() > 1:
                            loss_diff = net_reg.module.diff_loss(latents=latents_pred.detach(), prompt_embeds=prompt_embeds, )
                        else:
                            loss_diff = net_reg.diff_loss(latents=latents_pred.detach(), prompt_embeds=prompt_embeds,)
                        accelerator2.backward(loss_diff)
                        extra_loss["loss_diff"] = loss_diff
                        if accelerator2.sync_gradients:
                            accelerator2.clip_grad_norm_(net_reg.parameters(), args.max_grad_norm)
                        optimizer_reg.step()
                        optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)
                if global_step  > 1 and args.disc_start_iter >=0:
                    """
                    Discriminator loss: fake image vs real image
                    """
                    for _ in range(args.disc_freq):
                        # fake image
                        lossD_fake = net_disc(x=x_tgt_pred.detach(), for_real=False, for_disc=True).mean()
                        accelerator1.backward(lossD_fake)
                        # real image
                        lossD_real = net_disc(x=batch["x_tgt"].detach(),for_real=True, for_disc=True).mean()
                        accelerator1.backward(lossD_real)
                        extra_loss["lossD_fake"] = lossD_fake
                        extra_loss["lossD_real"] = lossD_real
                        if accelerator1.sync_gradients:
                            accelerator1.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                        optimizer_disc.step()
                        optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
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
                            # "train/concate_image": [wandb.Image(concate_image[idx].float().detach().cpu()*0.5+0.5, caption=f"idx={idx}") for idx in range(B)]
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu()*0.5+0.5, caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu()*0.5+0.5, caption=f"idx={idx}") for idx in range(B)],
                            "train/diff_pred": [wandb.Image(x_tgt_pred[idx].float().detach().cpu()*0.5+0.5, caption=f"idx={idx}") for idx in range(B)],
                            "train/gt_depth": [wandb.Image(gt_depth[idx], caption=f"idx={idx}") for idx in range(B)],
                            "train/pred_depth": [wandb.Image(pred_depth[idx], caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 10:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pt")
                        accelerator.unwrap_model(net_v2v).save_model(outf)
                        accelerator.unwrap_model(net_disc).save_model(outf.replace("model", "model_disc"))
            
                 
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
