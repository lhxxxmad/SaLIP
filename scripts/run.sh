# DATA_PATH=[Your ActivityNet data and videos path]
# apt-get install libsm6 libxext6
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
# split_hosts=($split_hosts)

# git remote set-url <your_url> https://<your_token>@github.com/<username>/<repo>.git

# MSRVTT --do_train 1 \
CUDA_VISIBLE_DEVICES=1 \
python3 -m torch.distributed.launch --nproc_per_node=1 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port 59999 \
main.py \
--do_train 1 \
--workers 0 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 8 \
--batch_size_val 256 \
--anno_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/MSRVTT/msrvtt_data \
--video_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/MSRVTT/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir outputs/msrvtt \
--embd_mode wti \
--do_gauss 0 \
--video_mask_rate 0.7 \
--text_mask_rate 0.7 \
--temp_loss_weight 1.0 \
--rec_loss_weight 1.0 \
--ret_loss_weight 1.0 \
--sal_predictor mlp \
--training_mask 0 \
--mask_mode mean \
# --init_model /mnt/bd/cxx-dataset/SaLIP/salip_mdoel/SaLIP_model/outputs/msrvtt_ViT-B-32/best.bin


# 

# MSRVTT demo
# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_train 1 \
# --workers 0 \
# --n_display 50 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/MSRVTT/msrvtt_data \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/MSRVTT/MSRVTT_Videos \
# --datatype msrvtt \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/msrvtt \
# --embd_mode wti \
# --do_gauss 1 \
# --sal_predictor ca+mlp \
# --training_mask 1 \
# --mask_mode mean \

# --init_model /mnt/bd/cxx-dataset/EMCL-Net/best_outputs/best.bin

# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${split_hosts[0]} \
# --master_port ${split_hosts[1]} \
# main.py \
# --do_train 1 \
# --workers 0 \
# --n_display 50 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/ActivityNet/activitynet_data \
# --video_path /mnt/bd/cxx-second/tal/raw_videos/anet/videos \
# --datatype activity \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --output_dir outputs \
# --embd_mode wti \
# --do_gauss 0 \

# --num_props 3
# --init_model /mnt/bd/cxx-dataset/EMCL-Net/outputs/pytorch_model.bin.best.1

# # MSVD
# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${split_hosts[0]} \
# --master_port ${split_hosts[1]} \
# main.py \
# --do_train 1 \
# --workers 0 \
# --n_display 50 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 8 \
# --batch_size_val 8 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSVD/msvd_data \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSVD/MSVD_Videos \
# --datatype msvd \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/msvd \
# --embd_mode wti \
# --do_gauss 0 \

# DiDeMo
# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_train 1 \
# --workers 0 \
# --n_display 50 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 8 \
# --batch_size_val 128 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/DiDeMo/data \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/DiDeMo/videos \
# --datatype didemo \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/didemo \
# --embd_mode wti \
# --do_gauss 1 \
# --video_mask_rate 0.7 \
# --text_mask_rate 0.7 \
# --temp_loss_weight 1.0 \
# --rec_loss_weight 1.0 \
# --ret_loss_weight 1.0 \
# --sal_predictor ca+mlp \
# --training_mask 1 \
# --mask_mode mean \
# --init_model /mnt/bd/cxx-dataset/SaLIP/SaLIP_model/outputs/msrvtt_ViT-B-32/best.bin

# anet
# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_eval 1 \
# --workers 0 \
# --n_display 50 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 8 \
# --batch_size_val 8 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/activitynet/anet \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/activitynet/anet/Activity_Videos \
# --datatype activity_grounding \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --output_dir outputs/activity \
# --embd_mode wti \
# --do_gauss 1 \
# --num_props 3 \
# --init_model /mnt/bd/cxx-third/GTVR/outputs/best/best.bin


# LSMDC 
# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_eval 1 \
# --workers 0 \
# --n_display 10 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 128 \
# --batch_size_val 128 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/LSMDC/Clip_LSMDC \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/LSMDC/Clip_LSMDC/LSMDC_Videos \
# --datatype lsmdc \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/lsmdc \
# --embd_mode wti \
# --do_gauss 1 \
# --mask_mode mean \
# --init_model /mnt/bd/cxx-dataset/SaLIP/SaLIP_model/outputs/msrvtt_ViT-B-32/best.bin


# CUDA_VISIBLE_DEVICES=0 \
# python3 -m torch.distributed.launch --nproc_per_node=1 \
# --master_addr ${ARNOLD_WORKER_0_HOST} \
# --master_port ${ARNOLD_WORKER_0_PORT} \
# main.py \
# --do_eval 1 \
# --workers 8 \
# --n_display 50 \
# --epochs 5 \
# --lr 1e-4 \
# --coef_lr 1e-3 \
# --batch_size 8 \
# --batch_size_val 128 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/activitynet/anet \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip_original/data/activitynet/anet/Activity_Videos \
# --datatype activity \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/activity \
# --embd_mode wti \
# --do_gauss 1 \
# --video_mask_rate 0.7 \
# --text_mask_rate 0.7 \
# --temp_loss_weight 1.0 \
# --rec_loss_weight 1.0 \
# --ret_loss_weight 1.0 \
# --sal_predictor ca+mlp \
# --training_mask 1 \
# --mask_mode mean \
# --init_model /mnt/bd/cxx-dataset/SaLIP/SaLIP_model/outputs/msrvtt_ViT-B-32/best.bin