CUDA_VISIBLE_DEVICES=4,5,6 python3.8 main.py \
    --mode single_all \
    --batch_size 17 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 20 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    --data_path ..\
    --debug
