CUDA_VISIBLE_DEVICES=2,7 python3.8 main.py \
    --mode single_point_ce \
    --dataset july22 \
    --batch_size 256 \
    --epoch 300 \
    --lr 0.001 \
    --src_len 64 \
    --embedding_dim 256 \
    --hidden_dim 512 \
    --valid_interval 1 \
    --data_path ../data \
    # --debug
