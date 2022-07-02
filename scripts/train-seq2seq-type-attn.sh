CUDA_VISIBLE_DEVICES=5,6,7,8 python3.8 main.py \
    --mode seq2seq_type_attn \
    --batch_size 2048 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 20 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    # --debug