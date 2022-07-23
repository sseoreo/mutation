CUDA_VISIBLE_DEVICES=6,7 python3.8 main.py \
    --mode single_point \
    --data_path ../11July \
    --dataset single_refine \
    --batch_size 256 \
    --epoch 100 \
    --lr 0.001 \
    --src_len 64 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --valid_interval 1 \
    # --debug