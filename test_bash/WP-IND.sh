#!/bin/bash

cd ../code/
python test.py --log_name WP-IND --exp ICML2025_reproduce --dataset_name WP-IND --test_epoch 1800 --setting Inductive \
                --dim 128 --num_init_layer 4 --num_layer 10 --num_head 4 