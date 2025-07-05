from moviepy.editor import VideoFileClip, ImageClip, clips_array
import os

# 定义输入视频文件路径
base_path = '/nvme0/public_data/Occupancy/proj/img2img-turbo/inputs/video_6_stage1_vsd2ssim1_video'
video_files = [
    'CAM_FRONT_LEFT.mp4',
    'CAM_FRONT_RIGHT.mp4',
    'CAM_LEFT_FRONT.mp4',
    'CAM_RIGHT_FRONT.mp4',
    'CAM_LEFT_BACK.mp4',
    'CAM_RIGHT_BACK.mp4',
    'CAM_BACK.mp4'
]

# 加载视频剪辑
clips = [VideoFileClip(os.path.join(base_path, file)) for file in video_files]

# 获取视频的宽度和高度
video_width = clips[0].size[0]
video_height = clips[0].size[1]

# 加载图片作为中间填充
image_path = '/nvme0/public_data/Occupancy/proj/img2img-turbo/inputs/20250519-185313.png'  # 替换为你的图片路径
image_clip = ImageClip(image_path).set_duration(clips[0].duration).set_fps(clips[0].fps)

# 调整图片尺寸以匹配视频
image_clip = image_clip.resize((video_width*0.7, video_height*2))

# 将图片分成上下两半
image_height = image_clip.size[1]
half_image_clip1 = image_clip.crop(y1=0, y2=image_height//2, x1=0, x2=image_clip.size[0])
half_image_clip2 = image_clip.crop(y1=image_height//2, y2=image_height, x1=0, x2=image_clip.size[0])

# 创建一个多画面布局
# 前面两个视频居中
front_clips = clips_array([[clips[0], clips[1]]])

# 后面一个视频居中
back_clip = clips[6]

# 侧面四个视频分布在两侧
side_clips = clips_array([[clips[2], half_image_clip1, clips[3]], [clips[4], half_image_clip2, clips[5]]])

# 合并所有视频
final_clip = clips_array([[front_clips], [side_clips], [back_clip]])

# 输出视频文件路径
output_file = os.path.join(base_path, 'combined_video.mp4')

# 保存合并后的视频
final_clip.write_videofile(output_file, codec='libx264')