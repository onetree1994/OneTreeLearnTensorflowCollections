import os
import struct
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def showImgs(imgs, col_num = 5, row_num = 5):
    plt.figure(figsize=(10,10))
    for i in range(col_num * row_num):
        plt.subplot(col_num,col_num,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i][:,:,0])
    plt.show()

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# 读取
train_x, train_y = load_mnist("/home/guoqing/prog/kaggle_dataset/mnist/guanfang", "train")
testa_x, testa_y = load_mnist("/home/guoqing/prog/kaggle_dataset/mnist/guanfang", "t10k")
# 拼接
X_train = np.vstack((train_x, testa_x))
Y_train = np.append(train_y, testa_y)
print(X_train.shape, Y_train.shape)
# 预处理
X_train = pd.DataFrame(X_train)
Y_train = pd.DataFrame(Y_train)
X_train = X_train / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes = 10)
print(X_train.shape, Y_train.shape)
