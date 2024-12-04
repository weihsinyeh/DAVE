#!/bin/bash
export CUDA_VISIBLE_DEVICES=`python -m get_gpu`
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python3 /tmp2/b10902118/DAVE/main.py \
--skip_train \
--data_path /tmp2/b10902118/fsc \
--model_path /tmp2/b10902118/DAVE/checkpoints \
--model_name DAVE_3_shot \
--backbone resnet50 \
--swav_backbone \
--reduction 8 \
--num_enc_layers 3 \
--num_dec_layers 3 \
--kernel_dim 3 \
--emb_dim 256 \
--num_objects 3 \
--num_workers 8 \
--use_query_pos_emb \
--use_objectness \
--use_appearance \
--batch_size 1 \
--pre_norm
