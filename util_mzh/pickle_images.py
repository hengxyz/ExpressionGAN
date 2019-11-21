import pickle
import shutil

def main():
    imgfiles = {'/data/zming/datasets/test/mtcnn_align_128/0/joseph.png',
           '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng.png'}
    duplicate_imgs(imgfiles, 6)

    obj = {'train': {'/data/zming/datasets/test/mtcnn_align_128/0/joseph_0.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng_0.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/joseph_1.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng_1.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/joseph_2.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng_2.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/joseph_3.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng_3.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/joseph_4.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng_4.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/joseph_5.png': 4,
                     '/data/zming/datasets/test/mtcnn_align_128/0/Zuheng_5.png': 4
                     }
           }
    #file = '../split/test_images.pickle'
    file = '../split/train_images.pickle'

    pickle.dump(obj, open(file,'w'))
    obj_read = pickle.load(open(file, 'rb'))

    return 0

def duplicate_imgs(imgfiles, duplicate):

    for i in range(duplicate):
        for img in imgfiles:
            img_str = str.split(img, '.')
            dst=(img_str[0] + '_%d.' + img_str[1]) % i
            shutil.copy(img, dst)
    return 0


if __name__ == '__main__':
    main()