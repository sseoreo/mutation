CUDA_VISIBLE_DEVICES=5,6 python3.8 main.py \
    --mode single_point_attn_ce \
    --dataset july22 \
    --batch_size 256 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 64 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    --data_path ../data \
    # --debug
