CUDA_VISIBLE_DEVICES=2,3 python3.8 main.py \
    --mode single_point_ce \
    --batch_size 1024 \
    --epoch 100 \
    --lr 0.0001 \
    --src_len 64 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    --data_path ../data
    # --debug
