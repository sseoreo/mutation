CUDA_VISIBLE_DEVICES=6,7 python3.8 main.py \
    --mode single_point_bce \
    --dataset july22 \
    --batch_size 128 \
    --epoch 300 \
    --lr 0.001 \
    --src_len 64 \
    --trg_len 1 \
    --embedding_dim 256 \
    --hidden_dim 512 \
    --train_interval 1 \
    --valid_interval 1 \
    --data_path ../data \
    # --debug
