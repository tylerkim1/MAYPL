#!/bin/bash

cd ../code/
python test.py --log_name WDv1 --exp ICML2025_reproduce --dataset_name WD20K100v1 --test_epoch 930 --setting Inductive \
                --dim 256 --num_init_layer 5 --num_layer 6 --num_head 32 --msg_add_tr