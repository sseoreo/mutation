CUDA_VISIBLE_DEVICES=4 python3.8 main.py \
    --mode single_point \
    --batch_size 7 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 10 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    --debug