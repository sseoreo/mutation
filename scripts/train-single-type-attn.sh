CUDA_VISIBLE_DEVICES=0,1,2,3 python3.8 main.py \
    --mode single_type_attn \
    --batch_size 4096 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 50 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    # --debug