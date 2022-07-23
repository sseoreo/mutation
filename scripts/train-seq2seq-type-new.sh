CUDA_VISIBLE_DEVICES=2 python3.8 main.py \
    --data_path ../11July \
    --mode seq2seq_type_new \
    --batch_size 1024 \
    --epoch 100 \
    --lr 0.0001 \
    --src_len 50 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --data_path data-refine\
    --valid_interval 1 \
    # --debug