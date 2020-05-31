#!/bin/sh



mkdir cifar10 2>/dev/zero

python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=cifar10
