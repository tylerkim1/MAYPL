#!/bin/bash

cd ../code/
python train.py --dataset_name WikiPeople--eval --setting Transductive --lr 1e-4 --dim 256 \
                --num_epoch 2900 --val_dur 2900 --num_init_layer 3 --num_layer 4 --num_head 32 \
                --early_stop 0 --batch_num 20 --train_graph_ratio 0.7 --model_dropout 0.1 --smoothing 0.0 \
                --log_name WP--eval --exp ICML2025_reproduce