import os
import arff
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import preprocessing

# set the directory of the dataset
file = open("data/DATASET.arff", 'r')




# def plot_confusion_matrix(cm, classes,
#                           title='Confusion matrix',
#                           cmap=plt.cm.jet):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val, labels):
    predictions = model.predict(x_val).argmax(axis=-1)
    truelabel = y_val.argmax(axis=-1)
    i = 0
    while i < len(predictions):
        print(predictions[i])
        print(truelabel[i])
        print('---------------')
        if predictions[i] >= 1:
            predictions[i] = 1
        if(truelabel[i] >= 1):
            truelabel[i] = 1
        i += 1
       # 将one-hot转化为label

    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')




def generate_model(shape):
    # 定义模型
    model = Sequential()

    model.add(Dense(22, input_dim=shape, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))
    print(model.summary())

    return model

#获取数据
def scrape_data():
    # 解码arff数据，文本标签转变为二进制数据
    decoder = arff.ArffDecoder()
    data = decoder.decode(file, encode_nominal=True)

    # 将原始数据分解为数据和标签
    vals = [val[0: -1] for val in data['data']]
    labels = [label[-1] for label in data['data']]

    for val in labels:
        if labels[val] != 0:
            labels[val] = 1

    #将标签和数据分成训练和验证集
    training_data = vals[0: int(.9 * len(vals))]
    training_labels = labels[0: int(.9 * len(vals))]
    validation_data = vals[int(.9 * len(vals)):]
    validation_labels = labels[int(.9 * len(vals)):]

    a = np.asarray(training_data, dtype=float)
    scaler = preprocessing.StandardScaler().fit(a)
    scaler.mean_
    scaler.var_
    scaler.scale_
    training_data = scaler.transform(a)

    b = np.asarray(validation_data, dtype=float)
    scaler = preprocessing.StandardScaler().fit(b)
    scaler.mean_
    scaler.var_
    scaler.scale_
    validation_data = scaler.transform(b)
    #print(training_labels)

    # 将原有的类别向量转换为独热编码的形式
    training_labels = to_categorical(training_labels, 5)
    validation_labels = to_categorical(validation_labels, 5)

    # 用numpy保存所有数组
    np.save('saved-files/vals', np.asarray(vals))
    np.save('saved-files/labels', np.asarray(labels))
    np.save('saved-files/training_data', np.asarray(training_data))
    np.save('saved-files/validation_data', np.asarray(validation_data))
    np.save('saved-files/training_labels', np.asarray(training_labels))
    np.save('saved-files/validation_labels', np.asarray(validation_labels))


scrape_data()


# 加载保存的数据
data_train = np.load('saved-files/training_data.npy')
label_train = np.load('saved-files/training_labels.npy')
data_eval = np.load('saved-files/validation_data.npy')
label_eval = np.load('saved-files/validation_labels.npy')

print(len(data_train[0]))
# 生成并编译模型
model = generate_model(len(data_train[0]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 初始化tensorboard
tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)

# 训练模型

history = model.fit(data_train, label_train, validation_data=(data_eval, label_eval), epochs=10, callbacks=[tensorboard])
loss_history = history.history["loss"]

numpy_loss_history = np.array(loss_history)
np.savetxt("saved-files/loss_history.txt", numpy_loss_history, delimiter=",")

#model = load_model('saved-files/model.h5')

# 评估模型的性能
print(model.evaluate(data_eval, label_eval))
print(model.evaluate(data_train, label_train))

#创建模型的图像:
plot_model(model, to_file='model.png', show_shapes=True)

plt.figure(1)

# 分析准确性
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# 分析loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('saved-files/model.h5')
# =========================================================================================
# 最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
# labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
# 比如这里我的labels列表
labels = ['Normal', 'Attack']

plot_confuse(model, data_eval, label_eval, labels)
# save the model for later so no retraining is needed



