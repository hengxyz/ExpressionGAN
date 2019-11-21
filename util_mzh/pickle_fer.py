import pickle
import shutil
from itertools import islice
import os
import numpy as np

def main():
    input = '/data/zming/datasets/fer2013/raw_182_160_png/'
    labelfile = '/data/zming/datasets/fer2013/emotionlabel.txt'
    #file = '../split/fer2013_images.pickle'
    file = '../split/fer2013_images_testastrain.pickle'

    images = os.listdir(input)
    images.sort()
    image_paths_flat = [os.path.join(input,image) for image in images]


    labels_flat = []
    usage_flat =  []
    with open(labelfile , 'r') as f:
        for line in islice(f, 1, None):
            strtmp = str.split(line)
            labels_flat.append(np.int(strtmp[1]))
            usage_flat.append(strtmp[2])

    index_train = [idx for idx, phrase in enumerate(usage_flat) if phrase == 'Training'] 
    index_test = [idx for idx, phrase in enumerate(usage_flat) if phrase == 'PublicTest']

    image_paths_train = [im for idx, im in enumerate(image_paths_flat) if idx in index_train]
    image_paths_test = [im for idx, im in enumerate(image_paths_flat) if idx in index_test]

    labels_train = [im for idx, im in enumerate(labels_flat) if idx in index_train]
    labels_test = [im for idx, im in enumerate(labels_flat) if idx in index_test]


    dict_train = {}
    for img, label in zip(image_paths_train, labels_train):
        dict_train[img]=label

    dict_test = {}
    for img, label in zip(image_paths_test, labels_test):
        dict_test[img]=label

    #obj = {'train': dict_train, 'test': dict_test}
    obj = {'train': dict_test, 'test': dict_train}


    pickle.dump(obj, open(file,'w'))
    obj_read = pickle.load(open(file, 'rb'))

    return 0



if __name__ == '__main__':
    main()
