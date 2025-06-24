#!/bin/bash

cd ../code/
python test.py --log_name wd-eval --exp ICML2025 --dataset_name wd50k-eval --test_epoch 3000 --setting Transductive \
                --dim 256 --num_init_layer 4 --num_layer 6 --num_head 16