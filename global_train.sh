#!/bin/bash


python ./train-controller/main_controller.py --dataset OULU --y_dim 6 --z_dim 50 --checkpoint_dir ./train-controller/controller_oulu_50z --save_dir ./train-controller/controller_oulu_50z --batch_size 60 --epoch 10

python ./joint-train/main.py --dataset OULU --y_dim 6 --checkpoint_dir ./train-controller/controller_oulu_50z --save_dir ./joint-train/oulu_50z --batch_size 48 --is_stage_one True --epoch 10
python ./joint-train/main.py --dataset OULU --y_dim 6 --checkpoint_dir ./joint-train/oulu_50z --save_dir ./joint-train/oulu_50z2 --batch_size 48 --is_stage_one False --epoch 10
