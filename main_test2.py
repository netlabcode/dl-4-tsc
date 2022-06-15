# Reff : https://github.com/Fdevmsy/Time-Series-Classification/blob/master/TimeSeriesCalssification.ipynb
# : https://github.com/abulbasar/neural-networks/blob/master/Keras%20-%20Multivariate%20time%20series%20classification%20using%20LSTM.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from sklearn import preprocessing
import tensorflow.keras
from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv1D
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.constants import CLASSIFIERS
from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets

data = pd.read_csv("creditcard.csv")

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)

def generate_data(X, y, sequence_length=10, step=1):
    X_local = []
    y_local = []
    for start in range(0, len(data) - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(y[end-1])
    return np.array(X_local), np.array(y_local)

X_sequence, y = generate_data(data.loc[:, "V1":"V28"].values, data.Class)

print(X_sequence.shape, y.shape)

output_directory = os.getcwd()
print(output_directory)

training_size = int(len(X_sequence) * 0.7)
x_train, y_train = X_sequence[:training_size], y[:training_size]
x_test, y_test = X_sequence[training_size:], y[training_size:]

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(x_train.shape) == 2:  # if univariate
# add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_train.shape[1:]

classifier = create_classifier('cnn', input_shape, nb_classes, output_directory)
classifier.fit(x_train, y_train, x_test, y_test, y_true)

print(input_shape)

# def fit_classifier():
#     x_train = datasets_dict[dataset_name][0]
#     y_train = datasets_dict[dataset_name][1]
#     x_test = datasets_dict[dataset_name][2]
#     y_test = datasets_dict[dataset_name][3]
#
#     nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
#
#     # transform the labels from integers to one hot vectors
#     enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
#     enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
#     y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
#     y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
#
#     # save orignal y because later we will use binary
#     y_true = np.argmax(y_test, axis=1)
#
#     if len(x_train.shape) == 2:  # if univariate
#         # add a dimension to make it multivariate with one dimension
#         x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
#         x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
#
#     input_shape = x_train.shape[1:]
#     classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
#
#     classifier.fit(x_train, y_train, x_test, y_test, y_true)

# model = tensorflow.keras.Sequential()
# model.add(LSTM(100, input_shape = (10, 28)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss="binary_crossentropy"
#               , metrics=[tensorflow.keras.metrics.binary_accuracy]
#               , optimizer="adam")

# model = tensorflow.keras.Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10,28)))
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
# model.add(Conv1D(filters=28, kernel_size=3, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(28))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss="binary_crossentropy"
#                , metrics=[tensorflow.keras.metrics.binary_accuracy]
#                , optimizer="adam")
#
# model.summary()

# training_size = int(len(X_sequence) * 0.7)
# X_train, y_train = X_sequence[:training_size], y[:training_size]
# X_test, y_test = X_sequence[training_size:], y[training_size:]
#
# model.fit(X_train, y_train, batch_size=64, epochs=50)
#
# model.evaluate(X_test, y_test)
# y_test_prob = model.predict(X_test, verbose=1)
# y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
# print(confusion_matrix(y_test, y_test_pred))