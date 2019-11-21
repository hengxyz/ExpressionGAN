#!/bin/bash

#python main.py --dataset OULU --y_dim 6 --checkpoint_dir controller_oulu_50z --save_dir oulu_50z --batch_size 48 --is_stage_one True --epoch 50
#python main.py --dataset OULU --y_dim 6 --checkpoint_dir oulu_50z --save_dir oulu_50z2 --batch_size 48 --is_stage_one False --epoch 50

#python ../train-controller/main_controller.py --dataset OULU --y_dim 6 --z_dim 50 --checkpoint_dir ../train-controller/controller_oulu_50z --save_dir ../joint-train/controller_oulu_50z --batch_size 60 --epoch 600
python main.py --dataset OULU --y_dim 6 --checkpoint_dir /data/zming/models/GAN/20180305-163443 --save_dir /data/zming/models/GAN --batch_size 48 --is_stage_one True --epoch 1000 --gpu_memory_fraction 0.9
python main.py --dataset OULU --y_dim 6 --checkpoint_dir oulu_50z --save_dir /data/zming/models/GAN --batch_size 48 --is_stage_one False --epoch 1000 --gpu_memory_fraction 0.9

##fer2013
python main.py --dataset /data/zming/datasets/fer2013/raw_182_160_png/ --y_dim 7 --checkpoint_dir /data/zming/models/GAN/20180311-020751 --save_dir /data/zming/models/GAN --batch_size 70 --is_stage_one True --epoch 1000 --split_file ../split/fer2013_images.pickle
