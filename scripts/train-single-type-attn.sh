CUDA_VISIBLE_DEVICES=6,7 python3.8 main.py \
    --mode single_type_attn \
    --batch_size 1024 \
    --epoch 100 \
    --lr 0.0001 \
    --src_len 50 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    # --debug