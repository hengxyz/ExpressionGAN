#!/bin/bash
#generating the expression images with the model trained on dataset oulucasia, the source images are from the dataset fer2013.

python gen_images.py --testing_sample_dir /data/zming/datasets/fer2013/raw_182_160_png --y_dim 6 --checkpoint_dir /data/zming/models/GAN/20180306-012727 --save_dir /data/zming/logs/GAN --batch_size 48 --is_stage_one False  --is_train False --generate_num 6144

python gen_images.py --testing_sample_dir /data/zming/datasets/Oulu_casia/OULU_128 --y_dim 6 --checkpoint_dir /data/zming/models/GAN/20180306-012727 --save_dir /data/zming/logs/GAN --batch_size 48 --is_stage_one False  --is_train False --generate_num 6144
