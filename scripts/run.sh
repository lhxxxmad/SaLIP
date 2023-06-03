# DATA_PATH=[Your ActivityNet data and videos path]
# apt-get install libsm6 libxext6
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
split_hosts=($split_hosts)

# MSRVTT --do_train 1 \
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
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSRVTT/msrvtt_data \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSRVTT/MSRVTT_Videos \
# --datatype msrvtt \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/msrvtt \
# --embd_mode wti \
# --do_gauss 1 \
# --interact_mode FGW \
# --sal_predictor trans
# --init_model /mnt/bd/cxx-dataset/EMCL-Net/best_outputs/msrvtt/best.bin
# --base_encoder None \


# 

# MSRVTT demo
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
# --batch_size 128 \
# --batch_size_val 4 \
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSRVTT/msrvtt_data \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/MSRVTT/MSRVTT_Videos \
# --datatype msrvtt \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/msrvtt \
# --embd_mode wti \
# --do_gauss 0 \
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

# --num_props 1
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
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/DiDeMo/data \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/DiDeMo/videos \
# --datatype didemo \
# --max_words 64 \
# --max_frames 64 \
# --video_framerate 1 \
# --output_dir outputs/didemo \
# --embd_mode wti \
# --do_gauss 0 \

# anet
CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 \
--master_addr ${split_hosts[0]} \
--master_port ${split_hosts[1]} \
main.py \
--do_train 1 \
--workers 0 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 8 \
--batch_size_val 8 \
--anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/activitynet/anet \
--video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/activitynet/anet/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir outputs/activity \
--embd_mode wti \
--do_gauss 1 \
# --init_model /mnt/bd/cxx-dataset/EMCL-Net/best_outputs/anet/best.bin


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
# --anno_path /mnt/bd/cxx-dataset/CLIP4Clip/data/LSMDC/Clip_LSMDC \
# --video_path /mnt/bd/cxx-dataset/CLIP4Clip/data/LSMDC/Clip_LSMDC/LSMDC_Videos \
# --datatype lsmdc \
# --max_words 32 \
# --max_frames 12 \
# --video_framerate 1 \
# --output_dir outputs/lsmdc \
# --embd_mode wti \
# --do_gauss 0 \
# --init_model /mnt/bd/cxx-dataset/EMCL-Net/best_outputs/best.bin