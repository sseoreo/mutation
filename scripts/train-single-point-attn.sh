CUDA_VISIBLE_DEVICES=5 python3.8 main.py \
    --mode single_point_attn_new_bce \
    --dataset july22 \
    --batch_size 256 \
    --epoch 300 \
    --lr 0.001 \
    --src_len 64 \
    --trg_len 1 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --train_interval 1 \
    --valid_interval 1 \
    --data_path ../data \
    --seed 371 \
    # --debug