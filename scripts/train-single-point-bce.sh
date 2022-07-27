CUDA_DEVICE=5

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3.8 main.py \
#     --mode single_point_bce \
#     --dataset july22 \
#     --batch_size 256 \
#     --epoch 300 \
#     --lr 0.001 \
#     --src_len 64 \
#     --trg_len 1 \
#     --embedding_dim 64 \
#     --hidden_dim 256 \
#     --train_interval 1 \
#     --valid_interval 1 \
#     --data_path ../data \
    # --debug


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3.8 main.py \
    --mode single_point_new_bce \
    --dataset july22 \
    --batch_size 256 \
    --epoch 300 \
    --lr 0.001 \
    --src_len 64 \
    --trg_len 1 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --train_interval 1 \
    --valid_interval 1 \
    --data_path ../data \
    # --debug
