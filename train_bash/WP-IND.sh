#!/bin/bash

cd ../code/
python train.py --dataset_name WP-IND --setting Inductive --lr 1e-4 --dim 128 \
                --num_epoch 1800 --val_dur 1800 --num_init_layer 4 --num_layer 10 --num_head 4 \
                --early_stop 0 --batch_num 50 --train_graph_ratio 0.7 --model_dropout 0.1 --smoothing 0.0 \
                --log_name WP-IND --exp ICML2025_reproduce