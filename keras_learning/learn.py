import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

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
X_train = train.sample(frac = 0.99)
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
print("after extract:")
print("X_train:\n", X_train.head(1))
print("X_valid:\n", X_valid.head(1))
# 绘制统计图表
g = sns.countplot(Y_train)
plt.show()
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
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
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
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
# 3.0.2 checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint("./keras_learning/mnist.hdf5")

# 3.1 训练
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=100),
                              epochs = 1, validation_data = (X_valid,Y_valid),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 100, 
                              callbacks=[learning_rate_reduction, early_stop, checkpoint])
# 4 分析训练结果
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# history = hist
# plt.plot(history["epoch"], history["loss"], color='b', label="train_loss")
# plt.plot(history["epoch"], history["val_loss"], color='r', label="val_loss")
# plt.show()
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()

# 还可以绘制 confusion matrix， 并显示到底是哪个错了
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 如果使用这个作弊训练出来的模型，会发现一点错误都没有了，就太假了
model = load_model("./keras_learning/weights.99.hdf5")
# Predict the values from the validation dataset
Y_pred = model.predict(X_valid)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_valid,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
plt.show()

# Display some error results 
# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_valid[errors]
def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]
# Show the top 6 errors
print(len(Y_pred_classes_errors),most_important_errors)
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# 5 预测并生成结果
results = model.predict(X_testa)
# select the indix with the maximum probability
results = np.argmax(results, axis = 1)
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
