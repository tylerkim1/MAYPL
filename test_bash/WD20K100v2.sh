#!/bin/bash

cd ../code/
python test.py --log_name WDv2 --exp MP_PQ --dataset_name WD20K100v2 --test_epoch 210 --setting Inductive \
                --dim 256 --num_init_layer 3 --num_layer 5 --num_head 8 --msg_add_tr