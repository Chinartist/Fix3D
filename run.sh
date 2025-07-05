
export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export TOKENIZERS_PARALLELISM=false
# export WANDB_MODE="offline"
# python -m debugpy --listen 4567 --wait-for-client src/inference_paired.py --model_name "" \

accelerate launch train_sd2i2i.py \
    --output_dir "outputs/sd2i2i/stage0" \
    --dataset_folder "/nvme0/public_data/Occupancy/proj/img2img-turbo/inputs/Pairdata" \
    --train_batch_size 4 \
    --num_training_epochs 1000 \
    --checkpointing_steps 1700 \
    --eval_freq 1700 \
    --num_samples_eval 50 \
    --gradient_accumulation_steps 1 \
    --viz_freq 25 \
    --report_to "wandb" --tracker_project_name "sd2i2i" \
    --enable_xformers_memory_efficient_attention \
    --dataloader_num_workers 4 \
    --learning_rate 2e-5 \
    --disc_start_iter -1 \
    --lambda_rec 1.0 \
    --lambda_ssim 0.0 \
    --lambda_lpips 1.0 \
    --lambda_depth 0.0 \
    --lambda_gan 0.0 \
    --lambda_vsd 0.0 \
    --lr_warmup_rate 0.0 \
    --mixed_precision "fp16" \
    --train_stage 0 \
    --pretrained_path /nvme0/public_data/Occupancy/proj/img2img-turbo/outputs/pix2pix_turbo/stage0_render/checkpoints/mix_sd2.pkl

# accelerate launch train_wanc2v.py \
#     --output_dir "outputs/wanc2v/stage0" \
#     --dataset_folder "/nvme0/public_data/Occupancy/proj/img2img-turbo/inputs/Multiview" \
#     --train_batch_size 2 \
#     --num_training_epochs 1000 \
#     --checkpointing_steps 1700 \
#     --eval_freq 1700 \
#     --num_samples_eval 50 \
#     --gradient_accumulation_steps 1 \
#     --viz_freq 25 \
#     --report_to "wandb" --tracker_project_name "C2V" \
#     --dataloader_num_workers 16 \
#     --learning_rate 2e-5 \
#     --lambda_rec 0.0 \
#     --lambda_ssim 0.0 \
#     --lambda_lpips 0.0 \
#     --lambda_depth 0.0 \
#     --lr_warmup_rate 0.0 \
#     --mixed_precision "bf16" \
#     --train_stage 0 \
#     --pretrained_path /nvme0/public_data/Occupancy/proj/img2img-turbo/outputs/wanc2v/stage0/checkpoints/model_8510.pkl

   
    