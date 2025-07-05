import os
import cv2
import glob
import numpy as np

def img2video(img_dir,out_dir):
    """
    Convert images in a directory to a video file.
    
    Parameters:
    - img_dir: Directory containing the images.
    - out_dir: Output directory for the video file.
    
    Returns:
    - None
    """
    
    # Get all image files in the directory
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    if not img_files:
        img_files = glob.glob(os.path.join(img_dir, '*.png'))         
    # Sort the image files
    img_files.sort()
    
    # Check if there are any images in the directory
    if not img_files:
        print("No images found in the directory.")
        return
    
    # Read the first image to get the dimensions
    frame = cv2.imread(img_files[0])
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object Mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 format
    os.makedirs(out_dir, exist_ok=True)  # Create output directory if it doesn't exist
    # Create output video file path
    out_file = os.path.join(out_dir, os.path.basename(img_dir) + '.mp4')
    
    # Create VideoWriter object
    out = cv2.VideoWriter(out_file, fourcc, 10.0, (width, height))
    
    # Loop through all images and write them to the video file
    for img_file in img_files:
        frame = cv2.imread(img_file)
        out.write(frame)
    
    # Release the VideoWriter object
    out.release()
    
    print(f"Video saved at {out_file}")
def img2video_ffmpeg(img_dir,out_dir,fps):
    """
    Convert images in a directory to a video file using ffmpeg.
    
    Parameters:
    - img_dir: Directory containing the images png or jpg.
    - out_dir: Output directory for the video file.
    - fps: Frames per second for the video. 
    """
    # Get all image files in the directory
    os.makedirs(out_dir, exist_ok=True)  # Create output directory if it doesn't exist
    out_file = os.path.join(out_dir, os.path.basename(img_dir) + '.mp4')

    # Use ffmpeg to create the video
    os.system(f"ffmpeg -framerate {fps} -i {os.path.join(img_dir, '%*.png')} -c:v libx264 -pix_fmt yuv420p {out_file}")

    print(f"Video saved at {out_file}")
    
if __name__ == "__main__":
    # Example usage
    img_dir = '/nvme0/public_data/Occupancy/proj/Generation/cosmos-transfer1/dataset/scene_edge'  # Replace with your image directory
    out_dir = '/nvme0/public_data/Occupancy/proj/Generation/cosmos-transfer1/dataset/scene_edge_video'  # Replace with your output directory
    # img2video_ffmpeg(img_dir, out_dir,5)
    for seg in os.listdir(img_dir):
        if os.path.isdir(os.path.join(img_dir,seg)):
            img2video_ffmpeg(os.path.join(img_dir,seg), out_dir,30)