from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  os
import shutil 
from glob import glob

def main():
    cnt = 0
    input_dir = '/data/zming/models/GAN/20180309-173921/samples'
    #output_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_182_160_1'
    
    imgs = glob(os.path.join(input_dir,'*.png'))
    for img in imgs:
        imgstrs = img.split('.')
        imgstrs = imgstrs[0].split('/')
       
        filename ='%05d'%int(imgstrs[-1])+'.png'
        os.rename(os.path.join(input_dir,img),os.path.join(input_dir, filename))

    return 0

if __name__ == '__main__':
    main()
