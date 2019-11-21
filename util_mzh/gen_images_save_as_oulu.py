from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  os
import shutil

EXPRSSIONS_TYPE =  ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise' ]

def main():
    tmpfiles = []
    keyframes = 3
    cnt = 0
    #emotion_labels_filename = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong.txt'
    # input_dir = '/data/zming/logs/GAN/20180310-233957/test'
    # output_dir = '/data/zming/datasets/Oulu-Casia/gen_images_1k_128'

    input_dir = '/data/zming/logs/GAN/20180312-011719/test'
    output_dir = '/data/zming/datasets/Oulu-Casia/gen_images_60k_128'

    # input_dir = '/data/zming/logs/GAN/20180312-130715/test'
    # output_dir = '/data/zming/datasets/Oulu-Casia/gen_images_180k_128'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    images = os.listdir(input_dir)

    # f.write('No.  Subjet   Expression   label   Image\n')
    for i, img in enumerate(images):
        img_str0 = str.split(img, '.')
        img_str = str.split(img_str0[0], '_')
        id = img_str[0]
        expr = EXPRSSIONS_TYPE[int(img_str[3])]
        id_folder = os.path.join(output_dir, id)
        expr_folder = os.path.join(output_dir, id, expr)
        if not os.path.exists(id_folder):
            os.mkdir(id_folder)
        if not os.path.exists(expr_folder):
            os.mkdir(expr_folder)

        imgdst = img_str0[0]+'.png'
        shutil.copy(os.path.join(input_dir,img), os.path.join(output_dir,id_folder,expr_folder, imgdst))
        print('Images %d'%i)
    return 0

if __name__ == '__main__':
    main()
