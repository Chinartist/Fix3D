import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
import numpy as np
import cv2
from transformers import CLIPImageProcessor
def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function

    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)

    parser.add_argument("--lambda_depth", default=1.0, type=float)
    parser.add_argument("--lambda_rec", default=1.0, type=float)
    parser.add_argument("--lambda_ssim", default=0.5, type=float)
    #lambda_vsd
    parser.add_argument("--lambda_vsd", default=1.0, type=float)

    #reg_freq
    parser.add_argument("--reg_freq", default=3, type=int)
    parser.add_argument("--disc_freq", default=1, type=int)
    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_path",type=str, default=None,)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--codebook_size", default=2048, type=int)
    # training details
    parser.add_argument("--disc_start_iter", type=int, default=30001, help="The iteration at which to start training the discriminator.")
    parser.add_argument("--train_stage", type=int, default=1,)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    #codebook_learning_rate
    parser.add_argument("--codebook_learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_rate", type=float, default=0.05, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    #neg_prompt
    parser.add_argument("--neg_prompt", type=str, default="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", help="The negative prompt to use for training.")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

import albumentations as A

def batch_random_crop(images, size):
    #the images should be the same size and shape (h,w,3)
    shape = images[0].shape
    h, w = shape[:2]
    assert h >= size[0] and w >= size[1], "Image size should be larger than crop size"
    x = random.randint(0, h - size[0])
    y = random.randint(0, w - size[1])
    cropped_images = []
    for img in images:
        cropped_img = img[x:x + size[0], y:y + size[1], :]
        cropped_images.append(cropped_img)
    return cropped_images
class RenderSRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split):

        super().__init__()
        gt_files  = os.path.join(dataset_folder, f"{split}_gt.txt")
        render_files  = os.path.join(dataset_folder, f"{split}_render.txt")
        with open (gt_files ,"r") as f:
            gt_datafiles = f.read().splitlines()
        with open (render_files ,"r") as f:
            render_datafiles = f.read().splitlines()
        self.render_datafiles =[x.strip() for x in render_datafiles if x.strip()]
        self.gt_datafiles =[x.strip() for x in gt_datafiles if x.strip()]
        self.ourscene_idx = []
        if split=="train":
            city_check = False
            ours_check = False
            nuscene_check = False
            realcar_check = False
            for i,gt in enumerate(self.gt_datafiles):
                assert os.path.exists(gt), f"File {gt} does not exist"
                if "Cityspaces" in gt:
                    city_check = True
                if "ourscene" in gt:
                    if i < len(self.render_datafiles):
                        self.ourscene_idx.append(i)
                    ours_check = True
                if "nuscene" in gt:
                    nuscene_check = True
                if "3DRealCar" in gt:
                    realcar_check = True
            # assert city_check or bdd_check, "The dataset should contain either city or bdd images"
            print("City check:", city_check)
            print("Ours check:", ours_check)
            print("Nuscene check:", nuscene_check)
            print("Realcar check:", realcar_check)
            print("our scene:", len(self.ourscene_idx))

        self.ft_rate = 0.5
        self.hq2lq=A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(allow_shifted=False, p=0.3),
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # A.ChromaticAberration(primary_distortion_limit= (-0.3, 0.3),secondary_distortion_limit= (-0.3, 0.3),mode='random',p=0.3),
            # A.Emboss(alpha=[0.5, 0.5], strength=[0.6, 0.7], p=0.3),
        ],
        )
        self.randomresizepad = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_LANCZOS4),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, fill=255),
        ],p=1.0)
        self.downsample_range = [1, 12]
    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.gt_datafiles)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        
        caption=""
        if "ourscene" not in self.gt_datafiles[idx] and  random.random() < self.ft_rate and len(self.ourscene_idx)>0:
            idx = random.choice(self.ourscene_idx)
        RandomScale =int(( random.random()*0.5+0.5)*1024)
        RandomResize = transforms.Compose([
            transforms.Resize(RandomScale, interpolation=transforms.InterpolationMode.LANCZOS),
            # transforms.CenterCrop(512),
        ])
        x_tgt = np.array(RandomResize(Image.open(self.gt_datafiles[idx])))

        if idx < len(self.render_datafiles):
            x_src = np.array(RandomResize(Image.open(self.render_datafiles[idx])))
            x_src, x_tgt = batch_random_crop([x_src, x_tgt], (512,512))
            scale = np.random.uniform(1, 2)
            w, h = x_src.shape[1], x_src.shape[0]
            x_src =cv2.resize(x_src, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_LANCZOS4)
            x_src = cv2.resize(x_src, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)
        else:
            x_tgt = batch_random_crop([x_tgt], (512,512))[0]
            x_src = x_tgt
            w, h = x_src.shape[1], x_src.shape[0]

            x_src = self.hq2lq(image= x_src)['image']
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            x_src =cv2.resize(x_src, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_LANCZOS4)
            x_src = cv2.resize(x_src, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)
        if "3DRealCar" in self.gt_datafiles[idx] and  idx < len(self.render_datafiles):
            car_scale = [ 128, 256,512]
            random_car_scale = random.choice(car_scale)
            randomresizepad = A.Compose([
            A.Resize(random_car_scale, random_car_scale, interpolation=cv2.INTER_LANCZOS4),
            A.PadIfNeeded(min_height=512, min_width=512, position="center",border_mode=cv2.BORDER_CONSTANT, fill=255),
        ],p=1.0)
            x_src = randomresizepad(image= x_src)['image']
            x_tgt =randomresizepad(image= x_tgt)['image']
        if random.random() > 0.5:
            x_tgt = cv2.flip(x_tgt, 1)
            x_src = cv2.flip(x_src, 1)
        # input images scaled to -1,1
        x_src = F.to_tensor(x_src)
        x_src = F.normalize(x_src, mean=[0.5], std=[0.5])
        # output images scaled to -1,1
        x_tgt = F.to_tensor(x_tgt)
        x_tgt = F.normalize(x_tgt, mean=[0.5], std=[0.5])
        assert x_src.shape == x_tgt.shape, f"Image shapes do not match: {x_src.shape} vs {x_tgt.shape}"
        # assert (x_src.shape[0], x_src.shape[1]) == (3, 512), f"Image shape is not (3, 512): {x_src.shape}"
        mask_rec = np.array(1)
        return {
            "x_tgt": x_tgt,
            "x_src": x_src,
            "mask_rec":torch.from_numpy(mask_rec)
        }

class RenderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split):

        super().__init__()
        gt_files  = os.path.join(dataset_folder, f"{split}_gt.txt")
        render_files  = os.path.join(dataset_folder, f"{split}_render.txt")
        with open (gt_files ,"r") as f:
            gt_datafiles = f.read().splitlines()
        with open (render_files ,"r") as f:
            render_datafiles = f.read().splitlines()
        self.render_datafiles =[x.strip() for x in render_datafiles if x.strip()]
        self.gt_datafiles =[x.strip() for x in gt_datafiles if x.strip()]
        self.wogt_idx = []
        if split=="train":
            for i in range(len(self.render_datafiles)):
                if i >= len(self.gt_datafiles):
                    self.wogt_idx.append(i)
    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.render_datafiles)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        
        caption=""
        # if   random.random() < 0.5  and idx < len(self.gt_datafiles):
        #     idx = random.choice(self.wogt_idx)
        RandomScale =int(( random.random()*0.5+0.5)*1024)
        RandomResize = transforms.Compose([
            transforms.Resize(RandomScale, interpolation=transforms.InterpolationMode.LANCZOS),
            # transforms.CenterCrop(512),
        ])
        

        if idx < len(self.gt_datafiles):
            x_tgt = np.array(RandomResize(Image.open(self.gt_datafiles[idx])))
            x_src = np.array(RandomResize(Image.open(self.render_datafiles[idx])))
            x_src, x_tgt = batch_random_crop([x_src, x_tgt], (512,512))
            scale = np.random.uniform(1, 2)
            w, h = x_src.shape[1], x_src.shape[0]
            x_src =cv2.resize(x_src, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_LANCZOS4)
            x_src = cv2.resize(x_src, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)
            mask_rec = np.array(1)
        else:
            x_src = np.array(RandomResize(Image.open(self.render_datafiles[idx])))
            # x_src[x_src.sum(-1) == 0] = 255  # fill empty pixels with white
            # x_tgt = np.zeros_like(x_src)
            x_src, x_tgt = batch_random_crop([x_src, x_src+0], (512,512))
            scale = np.random.uniform(1, 2)
            w, h = x_src.shape[1], x_src.shape[0]
            x_src =cv2.resize(x_src, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_LANCZOS4)
            x_src = cv2.resize(x_src, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)
            mask_rec = np.array(0)


 
        if random.random() > 0.5:
            x_tgt = cv2.flip(x_tgt, 1)
            x_src = cv2.flip(x_src, 1)
        # input images scaled to -1,1
        x_src = F.to_tensor(x_src)
        x_src = F.normalize(x_src, mean=[0.5], std=[0.5])
        # output images scaled to -1,1
        x_tgt = F.to_tensor(x_tgt)
        x_tgt = F.normalize(x_tgt, mean=[0.5], std=[0.5])
        assert x_src.shape == x_tgt.shape, f"Image shapes do not match: {x_src.shape} vs {x_tgt.shape}"
        # assert (x_src.shape[0], x_src.shape[1]) == (3, 512), f"Image shape is not (3, 512): {x_src.shape}"
 
        return {
            "x_tgt": x_tgt,
            "x_src": x_src,
            "mask_rec":torch.from_numpy(mask_rec)
        }

class Canny2ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.split = split
        image_dirs = os.path.join(dataset_folder, f"{split}.txt")
        with open(image_dirs, "r") as f:
            self.image_dirs = f.read().splitlines()
        self.image_dirs = [x.strip() for x in self.image_dirs if x.strip()]
    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.image_dirs)
    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        image_files = random.sample(glob(os.path.join(image_dir, "*.jpg")), 5)
        x_src = []
        x_cond = []
        for image_file in image_files:
            image = Image.open(image_file).convert("RGB")
            if image.size[0] > image.size[1]:
                image = image.resize((512, int(512 * image.size[1] / image.size[0])), Image.LANCZOS)
            else:
                image = image.resize((int(512 * image.size[0] / image.size[1]), 512), Image.LANCZOS)
            image = A.PadIfNeeded(min_height=512, min_width=512, position="random",border_mode=cv2.BORDER_CONSTANT, fill=255)(image=np.array(image))['image']
            
            canny = cv2.Canny(image, 100, 200)
            canny = canny[:, :, None]
            canny =255- np.concatenate([canny, canny, canny], axis=2)
            image = F.to_tensor(image)
            image = F.normalize(image, mean=[0.5], std=[0.5])

            canny = F.to_tensor(canny)
            canny = F.normalize(canny, mean=[0.5], std=[0.5])
            x_src.append(image)
            x_cond.append(canny)
        x_src = torch.stack(x_src, dim=1).clamp(-1.0, 1.0)
        x_cond = torch.stack(x_cond, dim=1).clamp(-1.0, 1.0)
        return {
            "x_src": x_src,
            "x_cond": x_cond,
        }
            



if __name__ == "__main__":
    dataset= Canny2ImageDataset(dataset_folder="/nvme0/public_data/Occupancy/proj/img2img-turbo/inputs/Multiview", split="train")
    x = dataset[0]
    print(x["x_src"].shape, x["x_cond"].shape)