#!/bin/bash

cd ../code/
python train.py --dataset_name MFB-IND --setting Inductive --lr 1e-3 --dim 128 \
                --num_epoch 1550 --val_dur 1550 --num_init_layer 3 --num_layer 3 --num_head 16 \
                --early_stop 0 --batch_num 20 --train_graph_ratio 0.7 --model_dropout 0.1 --smoothing 0.0 \
                --log_name MFB-IND --exp ICML2025_reproduce