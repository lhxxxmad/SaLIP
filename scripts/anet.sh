mkdir data
cd data
echo "download data"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/anet.tar.gz
tar -zxvf anet.tar.gz
cd ..

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/EMCL-Net/tvr/models/ViT-B-32.pt
mv ViT-B-32.pt ./tvr/models

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
split_hosts=($split_hosts)

DATA_PATH=./data/anet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
main.py \
--do_train 1 \
--do_eval 1 \
--workers 8 \
--n_display 5 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ \
--video_path ${DATA_PATH}/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir outputs/activity \
--embd_mode wti \
--do_gauss 1 \
--video_mask_rate 0.1 \
--text_mask_rate 0.1 \
--rec_trans_num_layers1 4 \
--rec_trans_num_layers2 4 \
--sal_predictor trans

echo "test model"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
main.py \
--do_eval 1 \
--workers 8 \
--n_display 5 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ \
--video_path ${DATA_PATH}/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--init_model outputs/activity/best.bin \
--output_dir outputs/activity \
--embd_mode wti \
--do_gauss 1 \
--video_mask_rate 0.1 \
--text_mask_rate 0.1 \
--rec_trans_num_layers1 4 \
--rec_trans_num_layers2 4 \
--tmp_trans_num_layers 4 \
--sal_predictor trans