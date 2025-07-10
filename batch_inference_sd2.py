import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from src.sd2_i2i import SD2I2I
from rich.progress import track
from rich import print
from tqdm import tqdm
from rapidocr import RapidOCR
from src.ram.models.ram_lora import ram
from src.ram import inference_ram as inference
def ocr_inpaint(src_img, target_img, result):
    if result.boxes is None:
        return target_img
    print(f"Detected {result.txts}")
    for box in result.boxes:
        bbox = [box[:,0].min(), box[:,1].min(), box[:,0].max(), box[:,1].max()]
        target_img[ int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = src_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    return target_img
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/nvme0/public_data/Occupancy/proj/Fix3D/inputs/JiaLei', help='path to the input image')
    parser.add_argument('--prompt', type=str, default='', help='the prompt to be used')
    parser.add_argument('--num_cycle', type=int, default=1, )
    parser.add_argument('--model_path', type=str, default='/nvme0/public_data/Occupancy/proj/Fix3D/outputs/sd2i2i/stage1/checkpoints/model_2010.pt', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='/nvme0/public_data/Occupancy/proj/Fix3D/inputs/JiaLei_post', help='the directory to save the output')
    parser.add_argument('--max_size', type=int, default=1000, help='minimum size of the input image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    os.makedirs(args.output_dir, exist_ok=True)
    # initialize the model
    model = SD2I2I( pretrained_path=args.model_path)
    engine = RapidOCR()
    model.set_eval()
    model.cuda()
    torch_dtype = torch.bfloat16 
    model.to(torch_dtype)
    # make sure that the input image is a multiple of 8
    if args.input_dir == '':
        raise ValueError('input_dir should be provided')
    if not os.path.exists(args.input_dir):
        raise ValueError('input_dir does not exist')
    input_dir = args.input_dir
    device = int(os.environ["CUDA_VISIBLE_DEVICES"])
    for seg in track(os.listdir(input_dir)):
        for input_image_path in tqdm(os.listdir(os.path.join(input_dir,seg))):
            input_image_path = os.path.join(input_dir,seg, input_image_path)
            if not os.path.isfile(input_image_path):
                continue
            if not input_image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            input_image = Image.open(input_image_path).convert('RGB')
            new_width, new_height = input_image.size
            if new_width > args.max_size or new_height > args.max_size:
                scale = min(args.max_size / new_width, args.max_size / new_height)
                new_width = int(new_width * scale)
                new_height = int(new_height * scale)
            new_width = new_width- new_width % 16
            new_height = new_height- new_height % 16
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(input_image_path)
            x_src = F.to_tensor(input_image).unsqueeze(0).cuda()
            x_src = x_src * 2 - 1
            x_src = x_src.to(dtype=torch_dtype)
            output_image = x_src 
            with torch.no_grad():
                batch = {
                    'x_src': output_image,
                }
                batch["prompt"] = [f'']
                output_image = model.infer(batch)
            output_image = output_image* 0.5 + 0.5
            output_pil = output_image[0].cpu()*255
            output_pil = output_pil.permute(1, 2, 0).byte().numpy()
            if 0:
                input_image = np.array(input_image)
                # result = engine(output_pil)
                # output_pil = ocr_inpaint(input_image, output_pil, result)
                # result = engine(input_image)
                # output_pil = ocr_inpaint(input_image, output_pil, result)
            output_pil = Image.fromarray(output_pil)
            # save the output image
            os.makedirs(os.path.join(args.output_dir,seg,),exist_ok=True)
            # os.makedirs(os.path.join(args.output_dir,seg,"gt"),exist_ok=True)
            output_pil.save(os.path.join(args.output_dir,seg, bname))
            # gt_image = Image.open(input_image_path.replace("renders","gt"))
            # gt_image.resize((output_pil.size[0]*2,output_pil.size[1]*2)).save(os.path.join(args.output_dir,seg,"gt", bname))
        # break
            # concate_image = Image.new('RGB', (input_image.width + output_pil.width, max(input_image.height, output_pil.height)))
            # concate_image.paste(input_image, (0, 0))
            # concate_image.paste(output_pil, (input_image.width, 0))
            # concate_image.save(os.path.join(args.output_dir, bname))
    
