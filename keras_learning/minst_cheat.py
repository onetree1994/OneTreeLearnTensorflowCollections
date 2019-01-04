import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import os
import struct

np.random.seed(2)

# 0. 工具函数
def showImgs(imgs, col_num = 5, row_num = 5):
    plt.figure(figsize=(10,10))
    for i in range(col_num * row_num):
        plt.subplot(col_num,col_num,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i][:,:,0])
    plt.show()

# 0. 加载最原始的mnist数据
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
X_train_pre = np.vstack((train_x, testa_x))
Y_train_pre = np.append(train_y, testa_y)
X_train_pre_dict = dict()
for col in range(len(X_train_pre[0])):
    X_train_pre_dict["pixel{}".format(col)] = X_train_pre[:, col]
X_train_pre = pd.DataFrame(X_train_pre_dict)
print(X_train_pre.shape, Y_train_pre.shape)
print(X_train_pre.head(1))

# 1. 加载数据并预处理
print("1. loaddata and preprocess...")
train = pd.read_csv("../kaggle_dataset/mnist/train.csv")
test = pd.read_csv("../kaggle_dataset/mnist/test.csv")
# 1.0 检查数据是否有问题
print(train.isnull().any().describe())
print(test.isnull().any().describe())
# 1.1 拆分train和validation
print("split train dataset...")
X_testa = test
X_train = train.sample(frac = 0.0001)
X_valid = train.drop(index = X_train.index)
print("train:", train.shape)
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
# 1.2 从数据中提取标签
print("before preprocess:")
print("X_train:\n", X_train.head(1))
print("X_valid:\n", X_valid.head(1))
Y_train = X_train["label"]
Y_valid = X_valid["label"]
X_train = X_train.drop(labels = ["label"], axis = 1)
X_valid = X_valid.drop(labels = ["label"], axis = 1)
X_train = pd.concat([X_train, X_train_pre])
Y_train = np.append(Y_train, Y_train_pre)
print("after extract:")
print("X_train:\n", X_train.shape)
print("X_valid:\n", X_valid.shape)
print("Y_train:\n", Y_train.shape)
# 1.3 归一化
X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_testa = X_testa / 255.0
print("after unitize:")
print("X_train:\n", X_train.head(1))
print("X_valid:\n", X_valid.head(1))
print("X_testa:\n", X_testa.head(1))
# 1.4 图像reshape 784->28*28
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_valid = X_valid.values.reshape(-1, 28, 28, 1)
X_testa = X_testa.values.reshape(-1, 28, 28, 1)
print("after reshape:")
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_testa:", X_testa.shape)
# 1.5 标签one-hot化
Y_train = to_categorical(Y_train, num_classes = 10)
Y_valid = to_categorical(Y_valid, num_classes = 10)
print("after onehotize:")
print("Y_train:\n", Y_train[1])
print("Y_valid:\n", Y_valid[1])
# 1.6 数据增强
datagen = ImageDataGenerator(
        featurewise_center=False,   # set input mean to 0 over the dataset
        samplewise_center=False,    # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.0, # Randomly zoom image 
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
datagen.fit(X_train)
# showImgs(X_train)

# 2. 构建模型
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# optimizer = tf.train.A
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# 3. 训练
# 3.0 设置callbacks
# 3.0.1 学习率衰减
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                             patience=3, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.00001)
def learnRate(epoch):
    lr = 0.001
    if (epoch+1.0) > 50:
        return lr / 50.0
    return lr / (epoch + 1)
learning_rate_reduction = tf.keras.callbacks.LearningRateScheduler(learnRate)
# 3.0.1 early stop
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
# 3.0.2 checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint("./keras_learning/weights.{epoch:02d}.hdf5", period=3)

# 3.1 训练
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=124),
                              epochs = 100, validation_data = (X_valid,Y_valid),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 124, 
                              callbacks=[learning_rate_reduction, early_stop, checkpoint])
# 4 预测并生成结果
results = model.predict(X_testa)
# select the indix with the maximum probability
results = np.argmax(results, axis = 1)
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
