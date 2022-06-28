# export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNTQ1YjE0YS0wNTczLTQyNWEtOTYyNi02ZjczNDM4NDZkMjkifQ=="
# neptune.init()
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --mode seq2seq_attn \
    --batch_size 1024 \
    --epoch 30 \
    --lr 0.0005 \
    --embedding_size 200 \
    --hidden_size 400 \
    --drop_p 0.5 \
    --valid_interval 1 \
    --num_workers 4 