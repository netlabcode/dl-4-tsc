from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
from classifiers.inception import Classifier_INCEPTION

import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import pandas as pd

import tensorflow.keras as keras
import tensorflow as tf
import time
from tensorflow.keras.callbacks import TensorBoard


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

def cnn(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

    input_layer = keras.layers.Input(input_shape)

    if input_shape[0] < 60:  # for italypowerondemand dataset
        padding = 'same'

    conv1 = keras.layers.Conv1D(filters=6, kernel_size=7, padding=padding, activation='sigmoid')(input_layer)
    conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

    conv2 = keras.layers.Conv1D(filters=12, kernel_size=7, padding=padding, activation='sigmoid')(conv1)
    conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

    flatten_layer = keras.layers.Flatten()(conv2)

    output_layer = keras.layers.Dense(units=nb_classes, activation='sigmoid')(flatten_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    print(model.summary())

    logs = os.path.join(output_directory, "logs/"+model_name)

    callback_log = TensorBoard(
        log_dir=logs,
        histogram_freq=0, write_graph=True, write_grads=False, write_images=False)

    history = model.fit(x=x_train, y=y_train,
                        epochs=epoch,
                        validation_data=(x_test, y_test),
                        callbacks=callback_log)

    model.save(output_directory+"/trained/"+model_name)

def resnet(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

    input_layer = keras.layers.Input(input_shape)

    n_feature_maps = 64

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    print(model.summary())

    logs = os.path.join(output_directory, "logs/" + model_name)

    callback_log = TensorBoard(
        log_dir=logs,
        histogram_freq=0, write_graph=True, write_grads=False, write_images=False)

    history = model.fit(x=x_train, y=y_train,
                        epochs=epoch,
                        validation_data=(x_test, y_test),
                        callbacks=callback_log)

    model.save(output_directory + "/trained/" + model_name)

def incp(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

    input_layer = keras.layers.Input(input_shape)

    verbose = False
    build = True
    batch_size = 64
    lr = 0.001,
    nb_filters = 32
    use_residual = True
    use_bottleneck = True
    depth = 6
    kernel_size = 41

    x = input_layer
    input_res = input_layer

    A = Classifier_INCEPTION( output_directory, input_shape, nb_classes)
    #model = Classifier_INCEPTION.build_model(input_shape,nb_classes)

    model = A.build_model(input_shape, nb_classes)


    print(model.summary())

    logs = os.path.join(output_directory, "logs/"+model_name)

    callback_log = TensorBoard(
        log_dir=logs,
        histogram_freq=0, write_graph=True, write_grads=False, write_images=False)

    history = model.fit(x=x_train, y=y_train,
                        epochs=epoch,
                        validation_data=(x_test, y_test),
                        callbacks=callback_log)

    model.save(output_directory+"/trained/"+model_name)

############################################### main

data = pd.read_csv("data/creditcard-small-2.csv")
output_directory = os.getcwd()
output_directory = os.path.join(output_directory,"out")

X_sequence, y = generate_data(data.loc[:, "V1":"V28"].values, data.Class)
#X_sequence, y = generate_data(data.loc[:, "V1"].values, data.Class)

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

epoch=10
#model_name = 'optimal-NN'
#cnn(output_directory,input_shape, nb_classes,epoch, "optimal", x_train, y_train, x_test, y_test)
#resnet(output_directory,input_shape, nb_classes,epoch, "resnet", x_train, y_train, x_test, y_test)
incp(output_directory,input_shape, nb_classes,epoch, "inception", x_train, y_train, x_test, y_test)

# classifier = create_classifier('inception', input_shape, nb_classes, output_directory)
# classifier.fit(x_train, y_train, x_test, y_test, y_true)
#
# classifier = create_classifier('resnet', input_shape, nb_classes, output_directory)
# classifier.fit(x_train, y_train, x_test, y_test, y_true)
#
# classifier = create_classifier('encoder', input_shape, nb_classes, output_directory)
# classifier.fit(x_train, y_train, x_test, y_test, y_true)
#
# classifier = create_classifier('fcn', input_shape, nb_classes, output_directory)
# classifier.fit(x_train, y_train, x_test, y_test, y_true)