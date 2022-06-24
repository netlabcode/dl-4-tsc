from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
from classifiers.inception import Classifier_INCEPTION
from classifiers.fcn import Classifier_FCN
from classifiers.mlp import Classifier_MLP
from classifiers.mcnn import Classifier_MCNN
from classifiers.mcdcnn import Classifier_MCDCNN

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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC, classification_report


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

def inception(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

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

def fcn(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

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

def mlp(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

    A = Classifier_MLP( output_directory, input_shape, nb_classes)

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

def mcdcnn(output_directory, input_shape, nb_classes, epoch, model_name, x_train, y_train, x_test, y_test, verbose=False, build=True):

    A = Classifier_MCDCNN( output_directory, input_shape, nb_classes)
    model = A.build_model(input_shape, nb_classes)


    print(model.summary())

    logs = os.path.join(output_directory, "logs/"+model_name)

    # callback_log = TensorBoard(
    #     log_dir=logs,
    #     histogram_freq=0, write_graph=True, write_grads=False, write_images=False)
    #
    # history = model.fit(x=x_train, y=y_train,
    #                     epochs=epoch,
    #                     validation_data=(x_test, y_test),
    #                     callbacks=callback_log)

    model.save(output_directory+"/trained/"+model_name)


############################################### main
def combineData():
    datas = pd.DataFrame()
    labels = pd.DataFrame()

    data1 = pd.read_csv("data/gcn/NO_error.csv")
    data2 = pd.read_csv("data/gcn/FA_error.csv")
    data3 = pd.read_csv("data/gcn/SE_error.csv")

    label1 = pd.read_csv("data/gcn/NO_label.csv")
    label1 = label1.iloc[:,0]
    label2 = pd.read_csv("data/gcn/FA_label.csv")
    label2 = label2.iloc[:,0]
    label3 = pd.read_csv("data/gcn/FA_label.csv")
    label3 = label3.iloc[:,0]

    datas = datas.append(data1)
    datas = datas.append(data2)
    datas = datas.append(data3)

    labels = labels.append(label1)
    labels = labels.append(label2)
    labels = labels.append(label3)

    combine = datas + labels

    combine.to_csv('combined2.csv', index=False, header=False)

data = pd.read_csv("data/combined_origin.csv")
print(data.shape)

output_directory = os.getcwd()
output_directory = os.path.join(output_directory,"out")

#X_sequence, y = generate_data(data.loc[:, "V1":"V25"].values, data.Class)
X_sequence, y = generate_data(data.iloc[:, :38].values, data.iloc[:,39])

training_size = int(len(X_sequence) * 0.7)
x_train, y_train = X_sequence[:training_size], y[:training_size]
x_test, y_test = X_sequence[training_size:], y[training_size:]



nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# Select all data
x_all = X_sequence
y_all = enc.transform(y.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(x_train.shape) == 2:  # if univariate
# add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_train.shape[1:]
epoch=100
print(input_shape, nb_classes)


print(x_all.shape, y_all.shape)


model_path = "out/trained/custom2"

model = keras.models.load_model(model_path)

model.summary()

y_pred_all = model.predict(x_all)

print(y_all.shape, y_pred_all.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(nb_classes):
    fpr[i], tpr[i], _ = roc_curve(y_all[:,i],y_pred_all[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_all.ravel(), y_pred_all.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= nb_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.figure()
lw = 2

plt.plot(fpr["micro"],tpr["micro"],label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),color="deeppink",linestyle=":",linewidth=4,)
#plot(fpr["macro"],tpr["macro"],label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),color="navy",linestyle=":",linewidth=4,)

# plt.plot( fpr[0], tpr[0], color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc[0],)
# plt.plot( fpr[1], tpr[1], lw=lw, label="ROC curve (area = %0.2f)" % roc_auc[1],)
# plt.plot( fpr[2], tpr[2], lw=lw, label="ROC curve (area = %0.2f)" % roc_auc[2],)
# plt.plot( fpr[3], tpr[3], lw=lw, label="ROC curve (area = %0.2f)" % roc_auc[3],)

plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.savefig('roc.png')

y_pred_all = np.argmax(y_pred_all, axis=1)
y_all = np.argmax(y_all, axis=1)

print(y_all.shape, y_pred_all.shape)

print(confusion_matrix(y_pred_all, y_all))



# visualizer = ROCAUC(model)
# visualizer.fit(x_train,y_train)
# visualizer.score(x_test,y_test)
# visualizer.show()

# oz = classification_report(model, x_train, x_train, X_test=x_test, y_test=y_test, support=True, is_fitted=True)
#
# print(oz)

#print(classification_report(y_test, y_pred))

# cnn(output_directory,input_shape, nb_classes,epoch, "cnn", x_train, y_train, x_test, y_test)
# resnet(output_directory,input_shape, nb_classes,epoch, "resnet", x_train, y_train, x_test, y_test)
# inception(output_directory,input_shape, nb_classes,epoch, "inception", x_train, y_train, x_test, y_test)
# fcn(output_directory,input_shape, nb_classes,epoch, "fcn", x_train, y_train, x_test, y_test)
# mlp(output_directory,input_shape, nb_classes,epoch, "mlp", x_train, y_train, x_test, y_test)


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