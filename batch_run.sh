#!/bin/bash

levels=(2 4 8 16 32 64 128 256)
for q in "${levels[@]}"
do
	export CUDA_VISIBLE_DEVICES=0
	python main.py --level $q --use_gpu 1 --batch_norm True --task hela --init_lr 0.001 --seed 1  
	python main.py --level $q --use_gpu 1 --batch_norm True --task hela --init_lr 0.001 --seed 2 
	python main.py --level $q --use_gpu 1 --batch_norm True --task hela --init_lr 0.001 --seed 3 
	python main.py --level $q --use_gpu 1 --batch_norm True --task hela --init_lr 0.001 --seed 4 
done

