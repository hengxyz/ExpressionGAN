from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  os
import shutil 
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imresize
import sys

def main():
    tmpfiles = []
    keyframes = 3
    cnt = 0
    imsize = 160
    #emotion_labels_filename = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong.txt'
    #input_dir = '/data/zming/datasets/Oulu-Casia/PreProcessImg/VL_Acropped/Strong'
    #input_dir = '../data/OULU'
    #output_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_182_160_1'
    #input_dir = '/data/zming/datasets/Oulu-Casia/gen_images_60k_128'
    #input_dir = '/data/zming/datasets/Oulu-Casia/gen_images_60k_160'
    input_dir = '/data/zming/datasets/Oulu-Casia/gen_images_180k_160'
    
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

    cnt = 0
    folders = os.listdir(input_dir)
    folders.sort()
    badfiles = []
    for sub in folders:
	if os.path.isdir(os.path.join(input_dir,sub)):
	    expressions = os.listdir(os.path.join(input_dir,sub))
	    expressions.sort()
	    for expre in expressions:
		imgs = [file for file in os.listdir(os.path.join(input_dir,sub,expre)) if file.endswith('.png')]
		for img in imgs:
                    try: 
		        im = imread(os.path.join(input_dir,sub,expre,img))
                        im = imresize(im, (imsize,imsize))
                        imsave(os.path.join(input_dir,sub,expre,img), im)
                        print('Saving image %d : %s'%(cnt, img))
                        cnt += 1
                    except IOError:
                        print('Image read IOError!============================================')
                        badfiles.append(os.path.join(input_dir,sub,expre,img))
                        os.remove(os.path.join(input_dir,sub,expre,img))

    print('%d files are bad with IOError!'%len(badfiles))
    for badfile in badfiles:      
        print(badfile)

    return 0

if __name__ == '__main__':
    main()
