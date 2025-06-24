#!/bin/bash

cd ../code/
python test.py --log_name WP --exp ICML2025_reproduce --dataset_name WikiPeople --test_epoch 2400 --setting Transductive \
                --dim 256 --num_init_layer 3 --num_layer 4 --num_head 32 