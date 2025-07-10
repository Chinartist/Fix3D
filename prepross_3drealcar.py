import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
import glob
def Clear_ExtraData(data_dir):
    fragments = os.listdir(data_dir)
    for frag in tqdm(fragments):
        try:
            time_fragments = os.listdir(os.path.join(data_dir, frag))
            for time_frag in tqdm(time_fragments,desc=frag):
                images = os.listdir(os.path.join(data_dir, frag, time_frag))
                images = [image for image in images if  "depth" in image or "conf" in image or "json" in image or "frame" not in image]
                for image in images:
                    os.remove(os.path.join(data_dir, frag, time_frag, image))
        except:
            print("Error: ",frag)
            continue
def Collect_Image(data_dir,output_dir):
    fragments = os.listdir(data_dir)
    for frag in tqdm(fragments):
        time_fragments = os.listdir(os.path.join(data_dir, frag))
        for time_frag in tqdm(time_fragments):
            images = os.listdir(os.path.join(data_dir, frag, time_frag))
            images = [image for image in images if image.endswith(".jpg") and "frame" in image]
            images.sort()
            for image in images:
                img = Image.open(os.path.join(data_dir, frag, time_frag, image))
                img.save(os.path.join(output_dir, frag+"-"+time_frag+"-"+image))
        #     break
        # break
    print("num of images: ",len(os.listdir(output_dir)))

def Generate_ImageList(output_dir,train_ratio=0.8):
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    files_shuf= random.sample(files, len(files))
    files_shuf_train = files_shuf[:int(len(files_shuf)*train_ratio)]
    files_shuf_val= files_shuf[int(len(files_shuf)*train_ratio):]
    with open(os.path.join(output_dir,"files.list"),"w") as f:
        for file in files:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"files_shuf.list"),"w") as f:
        for file in files_shuf:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"files_shuf_train.list"),"w") as f:
        for file in files_shuf_train:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"files_shuf_val.list"),"w") as f:
        for file in files_shuf_val:
            f.write(file+"\n")
    print("num of images: ",len(files))
    print("num of shuffled images: ",len(files_shuf))
    print("num of train images: ",len(files_shuf_train))
    print("num of test images: ",len(files_shuf_val))
def CollectGenerate_ImageList(data_dir,output_dir,train_ratio=0.8):
    fragments = os.listdir(data_dir)
    files = []
    for frag in tqdm(fragments):
        time_fragments = os.listdir(os.path.join(data_dir, frag))
        for time_frag in tqdm(time_fragments):
            images = os.listdir(os.path.join(data_dir, frag, time_frag))
            images = [image for image in images if image.endswith(".jpg") and "frame" in image]
            images.sort()
            for image in images:
                files.append(os.path.join(data_dir, frag, time_frag, image))
        #     break
        # break
    print("num of images: ",len(files))
    files_shuf= random.sample(files, len(files))
    files_shuf_train = files_shuf[:int(len(files_shuf)*train_ratio)]
    files_shuf_val= files_shuf[int(len(files_shuf)*train_ratio):]
    with open(os.path.join(output_dir,"files.list"),"w") as f:
        for file in files:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"files_shuf.list"),"w") as f:
        for file in files_shuf:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"files_shuf_train.list"),"w") as f:
        for file in files_shuf_train:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"files_shuf_val.list"),"w") as f:
        for file in files_shuf_val:
            f.write(file+"\n")
    print("num of images: ",len(files))
    print("num of shuffled images: ",len(files_shuf))
    print("num of train images: ",len(files_shuf_train))
    print("num of test images: ",len(files_shuf_val))
def CollectGenerate_ImageList_ByFrag(data_dir,output_dir,val_frag="600_800"):
    fragments = os.listdir(data_dir)
    files = []
    files_train = []
    files_val = []
    for frag in tqdm(fragments):
        time_fragments = os.listdir(os.path.join(data_dir, frag))
        for time_frag in tqdm(time_fragments):
            images = os.listdir(os.path.join(data_dir, frag, time_frag))
            images = [image for image in images if image.endswith(".jpg") and "frame" in image]
            images.sort()
            for image in images:
                files.append(os.path.join(data_dir, frag, time_frag, image))
                if frag==val_frag:
                    files_val.append(os.path.join(data_dir, frag, time_frag, image))
                else:
                    files_train.append(os.path.join(data_dir, frag, time_frag, image))
    files_shuf= random.sample(files, len(files))
    files_shuf_train = random.sample(files_train, len(files_train))
    files_shuf_val= random.sample(files_val, len(files_val))
    assert  len(files_shuf_train)+len(files_shuf_val)==len(files_shuf)==len(files)
    with open(os.path.join(output_dir,"all.txt"),"w") as f:
        for file in files:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"all_shuf.txt"),"w") as f:
        for file in files_shuf:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"train.txt"),"w") as f:
        for file in files_shuf_train:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"val.txt"),"w") as f:
        for file in files_shuf_val:
            f.write(file+"\n")
    print(f"{val_frag} is used for validation")
    print("num of images: ",len(files))
    print("num of shuffled images: ",len(files_shuf))
    print("num of train images: ",len(files_shuf_train))
    print("num of test images: ",len(files_shuf_val))
def CollectGenerate_ImageList_RenderPair(data_dir,output_dir,val_rate=0.1):
    gt_files = []
    render_files = []

    time_fragments = os.listdir(data_dir)
    for time_frag in tqdm(time_fragments):
        gt_images = os.listdir(os.path.join(data_dir, time_frag,"gt"))
        gt_images = [image for image in gt_images if image.endswith(".png")]
        gt_images.sort()
        render_images = os.listdir(os.path.join(data_dir, time_frag,"renders"))
        render_images = [image for image in render_images if image.endswith(".png")]
        render_images.sort()
        assert len(gt_images)==len(render_images)
        for i in range(len(gt_images)):
            gt_files.append(os.path.join(data_dir, time_frag,"gt", gt_images[i]))
            render_files.append(os.path.join(data_dir, time_frag,"renders", render_images[i]))
    train_gt_files = gt_files[:int(len(gt_files)*(1-val_rate))]
    train_render_files = render_files[:int(len(render_files)*(1-val_rate))]
    val_gt_files = gt_files[int(len(gt_files)*(1-val_rate)):]
    val_render_files = render_files[int(len(render_files)*(1-val_rate)):]
    assert len(train_gt_files)==len(train_render_files)
    assert len(val_gt_files)==len(val_render_files)
    with open(os.path.join(output_dir,"train_gt.txt"),"w") as f:
        for file in train_gt_files:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"train_render.txt"),"w") as f:
        for file in train_render_files:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"val_gt.txt"),"w") as f:
        for file in val_gt_files:
            f.write(file+"\n")
    with open(os.path.join(output_dir,"val_render.txt"),"w") as f:
        for file in val_render_files:
            f.write(file+"\n")
    print("num of gt images: ",len(gt_files))
    print("num of render images: ",len(render_files))
    print("num of train gt images: ",len(train_gt_files))
    print("num of train render images: ",len(train_render_files))
    print("num of val gt images: ",len(val_gt_files))
    print("num of val render images: ",len(val_render_files))
    
def Combine_Renders_Pairs(Render_dir,add_dir,output_dir,rate=3):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(Render_dir,"train_gt.txt"),"r") as f:
        train_gt_files = f.readlines()
    with open(os.path.join(Render_dir,"train_render.txt"),"r") as f:
        train_render_files = f.readlines()
    with open(os.path.join(Render_dir,"val_gt.txt"),"r") as f:
        val_gt_files = f.readlines()
    with open(os.path.join(Render_dir,"val_render.txt"),"r") as f:
        val_render_files = f.readlines()
    with open(os.path.join(add_dir,"train.txt"),"r") as f:
        train_add_files = f.readlines()
    with open(os.path.join(add_dir,"val.txt"),"r") as f:
        val_add_files = f.readlines()
    train_gt_files +=train_add_files[:int(len(train_gt_files)*rate)]
    with open(os.path.join(output_dir,"train_gt.txt"),"w") as f:
        for file in train_gt_files:
            f.write(file)
    with open(os.path.join(output_dir,"train_render.txt"),"w") as f:
        for file in train_render_files:
            f.write(file)
    with open(os.path.join(output_dir,"val_gt.txt"),"w") as f:
        for file in val_gt_files:
            f.write(file)
    with open(os.path.join(output_dir,"val_render.txt"),"w") as f:
        for file in val_render_files:
            f.write(file)
    print("num of gt images: ",len(train_gt_files))
    print("num of render images: ",len(train_render_files))
    print("num of val gt images: ",len(val_gt_files))
    print("num of val render images: ",len(val_render_files)) 

def Combine_Data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_gt_files = []
    train_render_files = []
    _3rc_dir = "/nvme0/public_data/Occupancy/proj/Fix3D/inputs/3DRealCar_Renders/image_pairs_output"
    _3rc_gt = []
    _3rc_render = []
    fragments = os.listdir(_3rc_dir)
    for frag in tqdm(fragments):
        _3rc_gt.extend(glob.glob(os.path.join(_3rc_dir, frag,"gt","*.png")))
        _3rc_render.extend(glob.glob(os.path.join(_3rc_dir, frag,"renders","*.png")))
    
    ourscene_dir = "/nvme0/public_data/Occupancy/proj/cache/Chinartist/ourscene_pair/ourscene_pair/"
    ourscene_gt = glob.glob(os.path.join(ourscene_dir,"gt","*.jpg"))
    ourscene_render_sharp = [gt.replace("gt","rendered_sharp").replace(".jpg",".png") for gt in ourscene_gt]
    ourscene_render_wosharp =[gt.replace("gt","rendered_wosharp").replace(".jpg",".png") for gt in ourscene_gt]

    train_gt_files = train_gt_files+_3rc_gt+ourscene_gt+ourscene_gt
    train_render_files = train_render_files+_3rc_render+ourscene_render_sharp+ourscene_render_wosharp
    # train_render_files = []
    # val_gt_files =train_gt_files_ourscene
    # val_render_files = train_render_files_ourscene
    with open(os.path.join(output_dir,"train_gt.txt"),"w") as f:
        for file in train_gt_files:
            f.write(file)
            f.write("\n")
    with open(os.path.join(output_dir,"train_render.txt"),"w") as f:
        for file in train_render_files:
            f.write(file)
            f.write("\n")
    # with open(os.path.join(output_dir,"val_gt.txt"),"w") as f:
    #     for file in val_gt_files:
    #         f.write(file)
    #         f.write("\n")
    # with open(os.path.join(output_dir,"val_render.txt"),"w") as f:
    #     for file in val_render_files:
    #         f.write(file)
    #         f.write("\n")
    print("num of gt images: ",len(train_gt_files))
    print("num of render images: ",len(train_render_files))
    # print("num of val gt images: ",len(val_gt_files[::-1]))
    # print("num of val render images: ",len(val_render_files[::-1])) 
def Get_MultiView_dir(output_dir,):
    os.makedirs(output_dir, exist_ok=True)
    _3rc_dir = "/nvme0/public_data/Occupancy/proj/cache/3DRealCar"
    _3rc_dirs = []
    fragments = os.listdir(_3rc_dir)
    for frag in tqdm(fragments):
        time_fragments = os.listdir(os.path.join(_3rc_dir, frag))
        for time_frag in tqdm(time_fragments):
            _3rc_dirs.append(os.path.join(_3rc_dir, frag, time_frag))
    _3rc_dirs.sort()
    print("3DRealCar directories: ",len(_3rc_dirs))
    with open(os.path.join(output_dir,"train.txt"),"w") as f:
        for dir in _3rc_dirs:
            f.write(dir+"\n")
            
if __name__ == "__main__":
    # Clear_ExtraData("/nvme0/public_data/Occupancy/data/3DRealCar/")
    output_dir = "/nvme0/public_data/Occupancy/proj/Fix3D/inputs/Pairdata/"
    # Get_MultiView_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    Combine_Data(output_dir)
