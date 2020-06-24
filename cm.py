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

model = load_model('saved-files/model.h5')
data_eval = np.load('saved-files/training_data.npy')
label_eval = np.load('saved-files/training_labels.npy')


file = open("data/final-dataset.arff", 'wr')
decoder = arff.ArffDecoder()
data = decoder.decode(file, encode_nominal=True)

#dataset = open("newdataset.arff", 'wr')
print(data[-1])

#
# # 将原始数据分解为数据和标签
# vals = [val[0: -1] for val in data['data']]
# labels = [label[-1] for label in data['data']]
# print(vals[0])
# print(vals[2])
# print(labels[0])
# print(labels[2])
# print(len(labels))
# i = 0
# count = 0
# while i < len(labels):
#     if labels[i] > 0:
#         count = count + 1
#     i = i+1
# print(count)
