#!/bin/bash

cd ../code/
python test.py --log_name FB-50 --exp ICML2025 --dataset_name FB-50 --test_epoch 320 --setting Inductive \
                --dim 32 --num_init_layer 4 --num_layer 5 --num_head 4 --msg_add_tr