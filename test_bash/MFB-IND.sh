#!/bin/bash

cd ../code/
python test.py --log_name MFB-IND --exp ICML2025_reproduce --dataset_name MFB-IND --test_epoch 1550 --setting Inductive \
                --dim 128 --num_init_layer 3 --num_layer 3 --num_head 16 