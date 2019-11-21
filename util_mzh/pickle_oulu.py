from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  os
import shutil 
### OULU-CASIA  ###
EXPRSSIONS_TYPE =  ['0=Neutral', '1=Anger', '2=Disgust', '3=Fear', '4=Happiness', '5=Sadness', '6=Surprise' ]

# def main():
#     tmpfiles = []
#     keyframes = 3
#     cnt = 0
#     emotion_labels_filename = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong.txt'
#     #input_dir = '/data/zming/datasets/Oulu-Casia/PreProcessImg/VL_Acropped/Strong'
#     input_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_182_160'
#     output_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_182_160_1'
#
#     i = 0
#     with open(emotion_labels_filename, 'w') as f:
#         folders = os.listdir(input_dir)
#         folders.sort()
#         #f.write('No.  Subjet   Expression   label   Image\n')
#         for sub in folders:
#             if os.path.isdir(os.path.join(input_dir,sub)):
#                 expressions = os.listdir(os.path.join(input_dir,sub))
#                 expressions.sort()
#                 for expre in expressions:
#
#                     #label = [x for x in EXPRSSIONS_TYPE if expre in x]
#                     #label = int(label[0][0])
#                     imgs = [file for file in os.listdir(os.path.join(input_dir,sub,expre)) if file.endswith('.png')]
#                     #imgs.sort()
#                     #f.write('%d   %s   %s   %d   %s\n'%(cnt, sub, 'Neutral', 0, os.path.join(input_dir,sub,expre,imgs[0])))
#                     #cnt += 1
#                     for img in imgs:
#                     #    f.write('%d   %s   %s   %d   %s\n' % (cnt, sub, expre, label, os.path.join(input_dir,sub,expre,imgs[-1-i])))
#                     #    cnt += 1
# 	                os.rename(os.path.join(input_dir,sub,expre,img),os.path.join(input_dir,sub,expre,sub+'_'+expre+'_'+img))
#     return 0

def main():
    tmpfiles = []
    keyframes = 3
    cnt = 0
    emotion_labels_filename = '/data/zming/datasets/Oulu-Casia/Emotion_labels_VIS_Strong.txt'
    #input_dir = '/data/zming/datasets/Oulu-Casia/PreProcessImg/VL_Acropped/Strong'
    input_dir = '../data/OULU'
    #output_dir = '/data/zming/datasets/Oulu-Casia/VL_Strong_mtcnnpy_182_160_1'

    i = 0
    with open(emotion_labels_filename, 'w') as f:
        folders = os.listdir(input_dir)
        folders.sort()
        #f.write('No.  Subjet   Expression   label   Image\n')
        for sub in folders:
            if os.path.isdir(os.path.join(input_dir,sub)):
                expressions = os.listdir(os.path.join(input_dir,sub))
                expressions.sort()
                for expre in expressions:

                    #label = [x for x in EXPRSSIONS_TYPE if expre in x]
                    #label = int(label[0][0])
                    imgs = [file for file in os.listdir(os.path.join(input_dir,sub,expre)) if file.endswith('.png')]
                    #imgs.sort()
                    #f.write('%d   %s   %s   %d   %s\n'%(cnt, sub, 'Neutral', 0, os.path.join(input_dir,sub,expre,imgs[0])))
                    #cnt += 1
                    for img in imgs:
                        imgstrs = img.split('.')
                        filename =imgstrs[-2]+'.jpeg'
                    #    f.write('%d   %s   %s   %d   %s\n' % (cnt, sub, expre, label, os.path.join(input_dir,sub,expre,imgs[-1-i])))
                    #    cnt += 1
	                os.rename(os.path.join(input_dir,sub,expre,img),os.path.join(input_dir,sub,expre,filename))
    return 0

if __name__ == '__main__':
    main()
