import os
import numpy as np
import random
from scipy.misc import imread, imresize
from PIL import Image

def generate_txt_label(ROOT_DIR = "./datasets/IJCAI_2019_AAAC_train"):
    with open("./labels.txt", "w") as f:
        for label in os.listdir(ROOT_DIR):
            SUBSET_DIR = os.path.join(ROOT_DIR, label)
            for imagename in os.listdir(SUBSET_DIR):
                f.writelines("{}\t{}\n".format(os.path.join(SUBSET_DIR, imagename), int(label)))

def preprocessor(im):
    image = imresize(im, [224, 224]).astype(np.float)
    image = ( image / 255.0 ) * 2.0 - 1.0
    return image
    
def load_path_label(fname=None, batch_shape=None, separator='\t', shuffle=True, onehot=False):
    batch_size = batch_shape[0]
    bn_classes = 110
    images = np.zeros(batch_shape, dtype=np.float32)
    if onehot:
        labels = np.zeros([batch_size, bn_classes], dtype=np.float32)
    else:
        labels = []
    idx = 0

    with open(fname, 'r') as f:
        lines = f.readlines()
        if shuffle:
            random.shuffle(lines)
        for x in lines:
            x = x.strip().split(separator)
            filepath = x[0]
            # with open(filepath, 'rb') as img:
                # raw_image = imread(img, mode='RGB')
            raw_image = Image.open(filepath).convert('RGB')
            image = preprocessor(raw_image)
            images[idx, :, :, :] = image
            if onehot:
                labels[idx, int(x[1])] = 1
            else:
                labels.append(int(x[1]))
            idx += 1
            if idx == batch_size:
                yield images, labels
                images = np.zeros(batch_shape)
                if onehot:
                    labels = np.zeros([batch_size, bn_classes])
                else:
                    labels = []
                idx = 0
        if idx > 0:
            yield images, labels

if __name__ == "__main__":
    generate_txt_label()
    #data = load_path_label("./labels.txt", batch_shape=[4, 224, 224, 3])
    #for images, labels in data:
    #    print(images, labels)
    #    import pdb; pdb.set_trace()
