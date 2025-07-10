
export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export TOKENIZERS_PARALLELISM=false
# export WANDB_MODE="offline"
# python -m debugpy --listen 4567 --wait-for-client src/inference_paired.py --model_name "" \

accelerate launch train_sd2i2i.py \
    --output_dir "outputs/sd2i2i/stage1" \
    --dataset_folder "/nvme0/public_data/Occupancy/proj/Fix3D/inputs/Pairdata" \
    --train_batch_size 4 \
    --num_training_epochs 1000 \
    --checkpointing_steps 2000 \
    --gradient_accumulation_steps 1 \
    --viz_freq 25 \
    --report_to "wandb" --tracker_project_name "sd2i2i" \
    --enable_xformers_memory_efficient_attention \
    --dataloader_num_workers 4 \
    --learning_rate 2e-5 \
    --disc_start_iter -1 \
    --lambda_rec 1.0 \
    --lambda_lpips 1.0 \
    --lambda_depth 0.0 \
    --lambda_gan 0.0 \
    --lambda_vsd 1.0 \
    --lr_warmup_rate 0.0 \
    --mixed_precision "bf16" \
    --train_stage 1 \
    --pretrained_path "/nvme0/public_data/Occupancy/proj/Fix3D/outputs/sd2i2i/stage0/checkpoints/model_6010.pt"
