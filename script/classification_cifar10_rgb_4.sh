#!/bin/bash

# 设置固定参数


cd ../real_world

# 循环不同的层数


echo "Running classification CIFAR-10_RGB Task"
python classification.py \
        --n_qubits 8 \
        --n_layers 18 \
        --max_layers 18 \
        --n_samples 600 \
        --n_test 1000 \
        --n_reps 4 \
        --n_epochs 500 \
        --n_repeats 10 \
        --data_type "cifar10_rgb" 



echo "All experiments completed!"