#
# Python script to train a NN to reproduce the normalised logarithm of the S1 oscillator strength of DHI dimers (log_f_S1_norm)
# D. Bosch and L. Blancafort, Universitat de Girona, April 2022
# Modified to include the electronic structure descriptors LS, AR and IN, tanh activation functions, and DHICA dataset, October 2025
#
# input: the fingerprints file as sys.argv[1] (QBF.out)
#
# output: NN_log_f_S1_norm_names.txt: names of dimers in validation and test sets
# NN_log_f_S1_norm_TDDFT_energies.txt: TDDFT energy of dimers in validation and test sets
# NN_log_f_S1_norm_predict.txt: predicted energy of dimers in validation and test sets
# compounds in the preceding three files follow the same order
# NN_log_f_S1_norm.txt: final value of training, validation and test loss
# NN_log_f_S1_norm.csv: history of training and validation losses for each epoch
#
# to fit the other G_rel_norm, E_Sn_norm, log_f_Sn_norm (n = 1, 2, 3) and AIQM_E_rel_norm endpoints change the name of the endpoint in line 34 ff
#
# see also the control part in lines 139 ff (CONTROL ARGUMENTS FOR THE DHI/DHICA DESCRIPTOR)
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

########## Enter here the property to be predicted ##########
endpoint = "log_f_S1_norm"
#############################################################

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "NN_%s" % endpoint
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
# variable y contains the endpoint data (log_f_S1_norm in this case)
#
y = dimer.filter(["%s" % endpoint])
x = dimer

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

names_train = x_train.filter(["Molec_name"])
names_test = x_test.filter(["Molec_name"])
names_val = x_val.filter(["Molec_name"])

x_train = x_train.drop(x_train.columns[0], axis=1)
x_test = x_test.drop(x_test.columns[0], axis=1)
x_val = x_val.drop(x_val.columns[0], axis=1)

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
x_train = x_train.drop("log_f_S1_norm", axis=1)
x_test = x_test.drop("log_f_S1_norm", axis=1)
x_val = x_val.drop("log_f_S1_norm", axis=1)
x_train = x_train.drop("log_f_S2_norm", axis=1)
x_test = x_test.drop("log_f_S2_norm", axis=1)
x_val = x_val.drop("log_f_S2_norm", axis=1)
x_train = x_train.drop("log_f_S3_norm", axis=1)
x_test = x_test.drop("log_f_S3_norm", axis=1)
x_val = x_val.drop("log_f_S3_norm", axis=1)
x_train = x_train.drop("G_rel_norm", axis=1)
x_test = x_test.drop("G_rel_norm", axis=1)
x_val = x_val.drop("G_rel_norm", axis=1)
x_train = x_train.drop("E_S1_norm", axis=1)
x_test = x_test.drop("E_S1_norm", axis=1)
x_val = x_val.drop("E_S1_norm", axis=1)
x_train = x_train.drop("E_S2_norm", axis=1)
x_test = x_test.drop("E_S2_norm", axis=1)
x_val = x_val.drop("E_S2_norm", axis=1)
x_train = x_train.drop("E_S3_norm", axis=1)
x_test = x_test.drop("E_S3_norm", axis=1)
x_val = x_val.drop("E_S3_norm", axis=1)
x_train = x_train.drop("AIQM_E_rel", axis=1)
x_test = x_test.drop("AIQM_E_rel", axis=1)
x_val = x_val.drop("AIQM_E_rel", axis=1)
x_train = x_train.drop("AIQM_E_rel_norm", axis=1)
x_test = x_test.drop("AIQM_E_rel_norm", axis=1)
x_val = x_val.drop("AIQM_E_rel_norm", axis=1)

#
# CONTROL ARGUMENTS FOR THE DHI/DHICA DESCRIPTOR
#
# comment out the dropping of the "DHI/DHICA" descriptor lines below when using the full QBF dataset (containing both DHI and DHICA)
#
x_train = x_train.drop("DHI/DHICA", axis=1)
x_test = x_test.drop("DHI/DHICA", axis=1)
x_val = x_val.drop("DHI/DHICA", axis=1)

x_train_post = x_train.drop("G_rel", axis=1)
x_test_post = x_test.drop("G_rel", axis=1)
x_val_post = x_val.drop("G_rel", axis=1)

tf.test.gpu_device_name()
tf.config.experimental.list_physical_devices(device_type='GPU')

model = keras.models.Sequential([
    keras.layers.Input(shape=x_train_post.shape[1:]),
    keras.layers.Dense(7, activation="tanh"),
    keras.layers.Dense(1)
])

hidden1 = model.layers[0]

model.get_layer('dense') is hidden1

model.summary()

weights, biases = hidden1.get_weights()

pd.set_option('display.max_rows', None)

model.compile(loss=root_mean_squared_error, optimizer=keras.optimizers.SGD(learning_rate=5e-3))
history = model.fit(x_train_post, y_train, epochs=500, validation_data=(x_val_post, y_val))

history.history.keys()

model.save_weights("previous_weights_1.weights.h5")

model2 = keras.models.Sequential([
    keras.layers.Input(shape=x_train_post.shape[1:]),
    keras.layers.Dense(7, activation="tanh"),
])

model2.load_weights("previous_weights_1.weights.h5")

model2.add(keras.layers.Dense(7, activation="tanh"))
model2.add(keras.layers.Dense(1, activation="tanh"))

model2.summary()

model2.compile(loss=root_mean_squared_error, optimizer=keras.optimizers.SGD(learning_rate=5e-3))
history = model2.fit(x_train_post, y_train, epochs=5000, validation_data=(x_val_post, y_val))
mse_test = model2.evaluate(x_test_post, y_test)
mse_test2 = model2.evaluate(x_train_post, y_train)
mse_test3 = model2.evaluate(x_val_post, y_val)
prediction_val = model2.predict(x_val_post)
prediction_test = model2.predict(x_test_post)

history.history.keys()
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.axis([0,5000,0,1])
save_fig("NN_%s" % endpoint)

model2.save("NN_%s.keras" % endpoint)
model2.save_weights("NN_%s.weights.h5" % endpoint)

with open('NN_%s_names.txt' % endpoint,'w') as namesfile:
    namesfile.write('%s\n' % (''.join(str(names_val))))
    namesfile.write('%s\n' % (''.join(str(names_test))))

with open('NN_%s.txt' % endpoint,'w') as testfile:
    testfile.write('%s\n' % (''.join(str(mse_test2))))
    testfile.write('%s\n' % (''.join(str(mse_test3))))
    testfile.write('%s\n' % (''.join(str(mse_test))))

with open('NN_%s_TDDFT_energies.txt' % endpoint,'w') as refenergies:
    refenergies.write('%s\n' % (''.join(str(y_val))))
    refenergies.write('%s\n' % (''.join(str(y_test))))

with open('NN_%s_predict.txt' % endpoint,'w') as predictfile:
    predictfile.write('%s\n' % 'Validation set')
    for el in prediction_val:
        predictfile.write('%s\n' % (' '.join(el.astype(str))))
    predictfile.write('%s\n' % 'Test set')
    for el in prediction_test:
        predictfile.write('%s\n' % (' '.join(el.astype(str))))
pd.DataFrame(history.history).to_csv("NN_%s.csv" % endpoint)
