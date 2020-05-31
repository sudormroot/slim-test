#!/bin/sh

script_path="`dirname $0`"

. $script_path/env.sh

export CUDA_VISIBLE_DEVICES=1

export UNIFIED_MEMORY_SET=no

stdbuf -oL python $script_path/ptb_word_lm.py --data_path=simple-examples/data/ --num_steps=15 --batch_size=20 2>&1 | tee $script_path/output_batchsize-20_umem-$UNIFIED_MEMORY_SET.txt
stdbuf -oL python $script_path/ptb_word_lm.py --data_path=simple-examples/data/ --num_steps=15 --batch_size=40 2>&1 | tee $script_path/output_batchsize-40_umem-$UNIFIED_MEMORY_SET.txt


export UNIFIED_MEMORY_SET=yes
stdbuf -oL python $script_path/ptb_word_lm.py --data_path=simple-examples/data/ --num_steps=15 --batch_size=20 2>&1 | tee $script_path/output_batchsize-20_umem-$UNIFIED_MEMORY_SET.txt
stdbuf -oL python $script_path/ptb_word_lm.py --data_path=simple-examples/data/ --num_steps=15 --batch_size=40 2>&1 | tee $script_path/output_batchsize-40_umem-$UNIFIED_MEMORY_SET.txt
