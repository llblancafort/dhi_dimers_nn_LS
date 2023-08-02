#
# Python script to train a NN to reproduce the relative energy of DHI dimers (G_rel)
# D. Bosch and L. Blancafort, Universitat de Girona, April 2022
# Modified to include the electronic structure descriptors LS, AR and IN, July 2023
#
# input: one of dhi_input_layer_generator.py output files as sys.argv[1] (QBS.out, QBB.out, QBF.out)
#
# output: NN_G_rel_names.txt: names of dimers in validation and test sets
# NN_G_rel_DFT_energies.txt: DFT energy of dimers in validation and test sets
# NN_G_rel_predict.txt: predicted energy of dimers in validation and test sets
# compounds in the preceding three files follow the same order
# NN_G_rel.txt: final value of training, validation and test loss
# NN_G_rel.csv: history of training and validation losses for each epoch
# 
# to control the descriptors, check lines 66 ff (CONTROL ARGUMENT FOR THE FITTED ENDPOINT)
# and 106 ff (CONTROL ARGUMENTS FOR THE ELECTRONIC STRUCTURE DESCRIPTORS)
#
import sys
import os
import tarfile
from six.moves import urllib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "NN_G_rel"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

dimerset = sys.argv[1]

dimer = pd.read_csv(dimerset, header = 0, sep="\t")

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

from tensorflow.python.keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
#
# CONTROL ARGUMENT FOR THE FITTED ENDPOINT
#
# variable y contains the endpoint data (G_rel in this case)
#
y = dimer.filter(["G_rel"])
x = dimer

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

names_train = x_train.filter(["Molec_name"])
names_test = x_test.filter(["Molec_name"])
names_val = x_val.filter(["Molec_name"])

x_train = x_train.drop(x_train.columns[0], axis=1)
x_test = x_test.drop(x_test.columns[0], axis=1)
x_val = x_val.drop(x_val.columns[0], axis=1)
#
x_train = x_train.drop("E_S1", axis=1)
x_test = x_test.drop("E_S1", axis=1)
x_val = x_val.drop("E_S1", axis=1)
x_train = x_train.drop("f_S1", axis=1)
x_test = x_test.drop("f_S1", axis=1)
x_val = x_val.drop("f_S1", axis=1)
x_train = x_train.drop("E_S2", axis=1)
x_test = x_test.drop("E_S2", axis=1)
x_val = x_val.drop("E_S2", axis=1)
x_train = x_train.drop("f_S2", axis=1)
x_test = x_test.drop("f_S2", axis=1)
x_val = x_val.drop("f_S2", axis=1)
x_train = x_train.drop("E_S3", axis=1)
x_test = x_test.drop("E_S3", axis=1)
x_val = x_val.drop("E_S3", axis=1)
x_train = x_train.drop("f_S3", axis=1)
x_test = x_test.drop("f_S3", axis=1)
x_val = x_val.drop("f_S3", axis=1)
x_train = x_train.drop("Ox_state", axis=1)
x_test = x_test.drop("Ox_state", axis=1)
x_val = x_val.drop("Ox_state", axis=1)
#
# CONTROL ARGUMENTS FOR THE ELECTRONIC STRUCTURE DESCRIPTORS
#
# the electronic structure descriptors are set by dropping the ones that shall not be included in the fitting
# in this case LS is included, and IN and AR are dropped
# to do the fitting without any electronic structure descriptor, you need to drop all three (LS, IN and AR)
#
#x_train = x_train.drop("LS", axis=1)
#x_test = x_test.drop("LS", axis=1)
#x_val = x_val.drop("LS", axis=1)
x_train = x_train.drop("IN", axis=1)
x_test = x_test.drop("IN", axis=1)
x_val = x_val.drop("IN", axis=1)
x_train = x_train.drop("AR", axis=1)
x_test = x_test.drop("AR", axis=1)
x_val = x_val.drop("AR", axis=1)

x_train_post = x_train.drop("G_rel", axis=1)
x_test_post = x_test.drop("G_rel", axis=1)
x_val_post = x_val.drop("G_rel", axis=1)

model = keras.models.Sequential([
    keras.layers.Input(shape=x_train_post.shape[1:]),
    keras.layers.Dense(7, activation="relu"),
    keras.layers.Dense(7, activation="relu"),
    keras.layers.Dense(1, activation="relu")
])

hidden1 = model.layers[0]

hidden2 = model.layers[1]

model.get_layer('dense') is hidden1

model.get_layer('dense') is hidden2

model.summary()

weights, biases = hidden1.get_weights()
weights, biases = hidden2.get_weights()

pd.set_option('display.max_rows', None)

model.compile(loss=root_mean_squared_error, optimizer=keras.optimizers.SGD(lr=1e-2))
history = model.fit(x_train_post, y_train, epochs=5000, validation_data=(x_val_post, y_val))
mse_test = model.evaluate(x_test_post, y_test)
mse_test2 = model.evaluate(x_train_post, y_train)
mse_test3 = model.evaluate(x_val_post, y_val)
prediction_val = model.predict(x_val_post)
prediction_test = model.predict(x_test_post)

history.history.keys()
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.axis([0,5000,0,70])
save_fig("NN_G_rel")
plt.show()

model.save("NN_G_rel.h5")
model.save_weights("NN_G_rel.ckpt")

provafile = open('NN_G_rel_names.txt','w')
provafile.write('%s\n' % (''.join(str(names_val))))
provafile.write('%s\n' % (''.join(str(names_test))))
provafile.close()

testfile = open('NN_G_rel.txt','w')
testfile.write('%s\n' % (''.join(str(mse_test2))))
testfile.write('%s\n' % (''.join(str(mse_test3))))
testfile.write('%s\n' % (''.join(str(mse_test))))
testfile.close()

reals = open('NN_G_rel_DFT_energies.txt','w')
reals.write('%s\n' % (''.join(str(y_val))))
reals.write('%s\n' % (''.join(str(y_test))))
reals.close()

predictfile = open('NN_G_rel_predict.txt','w')
predictfile.write('%s\n' % 'Validation set')
for el in prediction_val:
    predictfile.write('%s\n' % (' '.join(el.astype(str))))
predictfile.write('%s\n' % 'Test set')
for el in prediction_test:
    predictfile.write('%s\n' % (' '.join(el.astype(str))))
predictfile.close()
pd.DataFrame(history.history).to_csv("NN_G_rel.csv")
