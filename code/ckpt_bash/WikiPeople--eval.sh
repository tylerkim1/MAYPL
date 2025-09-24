#!/bin/bash

cd ../code/
python test.py --log_name WP--eval --exp ICML2025 --dataset_name WikiPeople--eval --test_epoch 2900 --setting Transductive \
                --dim 256 --num_init_layer 3 --num_layer 4 --num_head 32 