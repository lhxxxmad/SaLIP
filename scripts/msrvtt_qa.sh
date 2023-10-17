# cd ..
# hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/EMCL-Net/tvr/models/ViT-B-32.pt
# mv ViT-B-32.pt ./video_question_answering/modules

cd ./video_question_answering

# mkdir data
cd data/MSRVTT
echo "download data"
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/MSRVTT.tar.gz
tar -zxvf MSRVTT.tar.gz
cd ../..
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
split_hosts=($split_hosts)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
--nproc_per_node=8 \
main.py \
--do_train \
--num_thread_reader=8 \
--epochs=10 \
--batch_size=128 \
--n_display=50 \
--train_csv data/MSRVTT/train.jsonl \
--val_csv data/MSRVTT/test.jsonl \
--data_path data/MSRVTT/train_ans2label.json \
--features_path data/MSRVTT/MSRVTT/MSRVTT_Videos \
--output_dir ckpts/salip_msrvtt_qa_2 \
--lr 1e-4 --max_words 32 \
--max_frames 12 \
--batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d

hdfs dfs -put -f ckpts/salip_msrvtt_qa_2 hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/QA