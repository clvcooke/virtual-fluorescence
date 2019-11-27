#!/bin/bash

levels=(2 4 8 16 32 64 128)
for q in "${levels[@]}"
do
	strategy=learned
	task=pan
	export CUDA_VISIBLE_DEVICES=1
	python main.py --level $q --use_gpu 1 --batch_norm True --task $task --init_lr 0.005 --seed 2  --init_strategy $strategy 
	# python main.py --level $q --use_gpu 1 --batch_norm True --task $task --init_lr 0.001 --seed 2  --init_strategy $strategy 
	# python main.py --level $q --use_gpu 1 --batch_norm True --task $task --init_lr 0.001 --seed 3  --init_strategy $strategy
	# python main.py --level $q --use_gpu 1 --batch_norm True --task $task --init_lr 0.001 --seed 4  --init_strategy $strategy

done

