#!/bin/bash

cd ../code/
python train.py --dataset_name WD20K100v1 --setting Inductive --lr 1e-4 --dim 256 \
                --num_epoch 930 --val_dur 930 --num_init_layer 5 --num_layer 6 --num_head 32 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.5 --model_dropout 0.05 --smoothing 0.0 \
                --log_name WDv1 --exp ICML2025_reproduce --msg_add_tr