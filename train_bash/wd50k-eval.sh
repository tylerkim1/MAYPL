#!/bin/bash

cd ../code/
python train.py --dataset_name wd50k-eval --setting Transductive --lr 1e-4 --dim 256 \
                --num_epoch 3000 --val_dur 3000 --num_init_layer 4 --num_layer 6 --num_head 16 \
                --early_stop 0 --batch_num 20 --train_graph_ratio 0.7 --model_dropout 0.2 --smoothing 0.05 \
                --log_name wd-eval --exp ICML2025_reproduce