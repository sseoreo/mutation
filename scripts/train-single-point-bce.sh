CUDA_VISIBLE_DEVICES=5,6 python3.8 main.py \
    --mode single_point_bce \
    --dataset july22 \
    --batch_size 256 \
    --epoch 300 \
    --lr 0.001 \
    --src_len 64 \
    --embedding_dim 256 \
    --hidden_dim 512 \
    --train_interval 10 \
    --valid_interval 1 \
    --data_path ../data \
    # --debug
