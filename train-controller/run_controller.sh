#!/bin/bash

#python main_controller.py --dataset OULU --y_dim 6 --z_dim 50 --checkpoint_dir controller_oulu_50z --save_dir controller_oulu_50z --batch_size 60
python main_controller.py --dataset OULU --y_dim 6 --z_dim 50 --checkpoint_dir controller_oulu_50z --save_dir controller_oulu_50z --batch_size 60 --epoch 600

###fer2013
 python main_controller.py --dataset /data/zming/datasets/fer2013/raw_182_160_png/ --y_dim 7 --checkpoint_dir /data/zming/models/GAN/ --save_dir /data/zming/models/GAN --batch_size 70 --is_stage_one False --epoch 200 --split_file ../split/fer2013_images_testastrain.pickle