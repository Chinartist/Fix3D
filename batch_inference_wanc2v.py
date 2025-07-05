import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from src.wan_c2v import WanC2V
from rich.progress import track
from rich import print
from tqdm import tqdm
import cv2
import albumentations as A
def resize_and_pad(image, target_size=512):
    if image.size[0] > image.size[1]:
        image = image.resize((target_size, int(target_size * image.size[1] / image.size[0])), Image.LANCZOS)
    else:
        image = image.resize((int(target_size * image.size[0] / image.size[1]), target_size), Image.LANCZOS)
    image = A.PadIfNeeded(min_height=target_size, min_width=target_size, position="center",border_mode=cv2.BORDER_CONSTANT, fill=255)(image=np.array(image))['image']
    return Image.fromarray(image)
def canny_from_pil(image, low_threshold=100, high_threshold=200):
    ori_image = np.array(image)[..., :3]
    # print(ori_image.shape)
    image = cv2.Canny(ori_image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = 255-np.concatenate([image, image, image], axis=2)
    # image = np.concatenate([image,ori_image], axis=1)
    control_image = Image.fromarray(image)
    return control_image
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default='/nvme0/public_data/Occupancy/proj/Pose_Free_Rec/data/Four_View_sr_masked', help='path to the input image')
    parser.add_argument('--input_dir', type=str, default='/nvme0/public_data/Occupancy/proj/Pose_Free_Rec/data/images', help='path to the input image')

    parser.add_argument('--model_path', type=str, default='/nvme0/public_data/Occupancy/proj/img2img-turbo/outputs/wanc2v/stage0/checkpoints/model_6810.pkl', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='/nvme0/public_data/Occupancy/proj/Pose_Free_Rec/data/images_gen', help='the directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # initialize the model
    model = WanC2V( pretrained_path=args.model_path)
    model.scheduler.set_timesteps(50, device=torch.device("cuda"))
    model.requires_grad_(False)
    model.eval()    
    model.cuda()
    dtype=torch.bfloat16
    model.to(dtype)

    input_dir = args.input_dir
    ref_dir = args.ref_dir
    for input_image_path in tqdm(os.listdir(input_dir)):
        input_image_path = os.path.join(input_dir, input_image_path)
        
        input_image = Image.open(input_image_path).convert('RGB')
        input_image = resize_and_pad(input_image)
        input_image_canny = canny_from_pil(input_image)
        bname = os.path.basename(input_image_path)
        x_src = []
        x_cond = [F.to_tensor(input_image_canny)]
        for ref_image_path in os.listdir(ref_dir):
            ref_image_path = os.path.join(ref_dir, ref_image_path)
            ref_image = Image.open(ref_image_path).convert('RGB')
            ref_image = resize_and_pad(ref_image)
            ref_image_canny = canny_from_pil(ref_image)
            x_src.append(F.to_tensor(ref_image))
            x_cond.append(F.to_tensor(ref_image_canny))
        x_src = torch.stack(x_src, dim=1).cuda().to(dtype)*2-1
        x_cond = torch.stack(x_cond, dim=1).cuda().to(dtype)*2-1
        output_image= model.infer({"x_src": x_src[None], "x_cond": x_cond[None],"prompt":[""]})
        
        output_image = output_image* 0.5 + 0.5
        output_pil = output_image[0].cpu()*255
        output_pil = output_pil.permute(1, 2, 0).byte().numpy()
        
        output_pil = Image.fromarray(output_pil)
        # save the output image
        os.makedirs(args.output_dir,exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir,seg,"gt"),exist_ok=True)
        output_pil.save(os.path.join(args.output_dir, bname))
        
    
