from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  os
import shutil

def main():
    tmpfiles = []
    keyframes = 3
    cnt = 0
    #emotion_labels_filename = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong.txt'
    input_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_128/'
    #input_dir = '../data/OULU'
    output_dir = '/data/zming/datasets/Oulu-Casia/id_images'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    folders = os.listdir(input_dir)
    folders.sort()
    # f.write('No.  Subjet   Expression   label   Image\n')
    for id in folders:
        expres = os.listdir(os.path.join(input_dir, id))
        imgs = os.listdir(os.path.join(input_dir, id, expres[0]))
        shutil.copy(os.path.join(input_dir,id,expres[0], imgs[0]), os.path.join(output_dir,id+'_'+expres[0]+'_'+imgs[0]))
        print('Person %s'%id)
    return 0

if __name__ == '__main__':
    main()
