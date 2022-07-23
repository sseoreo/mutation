CUDA_VISIBLE_DEVICES=0,1 python3.8 main.py \
    --mode seq2seq_point \
    --batch_size 1024 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 64 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    # --data_path ..\
    # --debug
