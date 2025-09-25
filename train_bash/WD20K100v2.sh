#!/bin/bash

cd ../code/
python train.py --dataset_name WD20K100v2 --setting Inductive --lr 2e-4 --dim 256 \
                --num_epoch 200 --val_dur 200 --num_init_layer 3 --num_layer 5 --num_head 8 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.4 --model_dropout 0.05 --smoothing 0.0 \
                --log_name WDv2 --exp Curriculum --msg_add_tr --curriculum