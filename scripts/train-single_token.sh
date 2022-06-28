CUDA_VISIBLE_DEVICES=6,7 python3.8 main.py \
    --mode single_token \
    --batch_size 2048 \
    --epoch 200 \
    --lr 0.001 \
    --embedding_size 64 \
    --hidden_size 64 \
    --valid_interval 1 \
    --seed 8 \
    # --debug