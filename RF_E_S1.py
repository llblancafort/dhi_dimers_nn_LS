#
# Python script to train a RF to reproduce the S1 vertical excitation of DHI dimers (E_S1)
# D. Bosch and L. Blancafort, Universitat de Girona, October 2025
#
# input: the fingerprints file as sys.argv[1] (QBF.out)
#
# output: RF_E_S1_names.txt: names of dimers in test set
# RF_E_S1_TDDFT_energies.txt: TDDFT energy of dimers in test set
# RF_E_S1_predict.txt: predicted energy of dimers in test set
# compounds in the preceding three files follow the same order
#
# to fit the other G_rel, E_Sn and f_Sn (n = 1, 2, 3) and AIQM_E_rel endpoints change the name of the endpoint in line 33 ff
#
# see also the control part in lines 118 ff (CONTROL ARGUMENTS FOR THE DHI/DHICA DESCRIPTOR)
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
from sklearn.ensemble import RandomForestRegressor

########## Enter here the property to be predicted ##########
endpoint = "E_S1"
#############################################################

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "RF_%s" % endpoint
#IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
#os.makedirs(IMAGES_PATH, exist_ok=True)

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
# variable y contains the endpoint data (E_S1 in this case)
#
y = dimer.filter(["%s" % endpoint])
x = dimer

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

names_train = x_train.filter(["Molec_name"])
names_test = x_test.filter(["Molec_name"])

x_train = x_train.drop(x_train.columns[0], axis=1)
x_test = x_test.drop(x_test.columns[0], axis=1)

x_train = x_train.drop("E_S1", axis=1)
x_test = x_test.drop("E_S1", axis=1)
x_train = x_train.drop("f_S1", axis=1)
x_test = x_test.drop("f_S1", axis=1)
x_train = x_train.drop("E_S2", axis=1)
x_test = x_test.drop("E_S2", axis=1)
x_train = x_train.drop("f_S2", axis=1)
x_test = x_test.drop("f_S2", axis=1)
x_train = x_train.drop("E_S3", axis=1)
x_test = x_test.drop("E_S3", axis=1)
x_train = x_train.drop("f_S3", axis=1)
x_test = x_test.drop("f_S3", axis=1)
x_train = x_train.drop("Ox_state", axis=1)
x_test = x_test.drop("Ox_state", axis=1)
x_train = x_train.drop("log_f_S1_norm", axis=1)
x_test = x_test.drop("log_f_S1_norm", axis=1)
x_train = x_train.drop("log_f_S2_norm", axis=1)
x_test = x_test.drop("log_f_S2_norm", axis=1)
x_train = x_train.drop("log_f_S3_norm", axis=1)
x_test = x_test.drop("log_f_S3_norm", axis=1)
x_train = x_train.drop("G_rel_norm", axis=1)
x_test = x_test.drop("G_rel_norm", axis=1)
x_train = x_train.drop("E_S1_norm", axis=1)
x_test = x_test.drop("E_S1_norm", axis=1)
x_train = x_train.drop("E_S2_norm", axis=1)
x_test = x_test.drop("E_S2_norm", axis=1)
x_train = x_train.drop("E_S3_norm", axis=1)
x_test = x_test.drop("E_S3_norm", axis=1)
x_train = x_train.drop("AIQM_E_rel", axis=1)
x_test = x_test.drop("AIQM_E_rel", axis=1)
x_train = x_train.drop("AIQM_E_rel_norm", axis=1)
x_test = x_test.drop("AIQM_E_rel_norm", axis=1)

#
# CONTROL ARGUMENTS FOR THE DHI/DHICA DESCRIPTOR
#
# comment out the dropping of the "DHI/DHICA" descriptor lines below when using the full QBF dataset (containing both DHI and DHICA)
#
x_train = x_train.drop("DHI/DHICA", axis=1)
x_test = x_test.drop("DHI/DHICA", axis=1)

x_train_post = x_train.drop("G_rel", axis=1)
x_test_post = x_test.drop("G_rel", axis=1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

tf.test.gpu_device_name()
tf.config.experimental.list_physical_devices(device_type='GPU')

forest_reg = RandomForestRegressor(max_depth=8, max_features=18, max_leaf_nodes=150, n_estimators=100)
forest_reg.fit(x_train_post, y_train.values.ravel())

prediction_train = forest_reg.predict(x_train_post)
prediction_test = forest_reg.predict(x_test_post)

with open('RF_%s_names.txt' % endpoint,'w') as namesfile:
    namesfile.write('%s\n' % (''.join(str(names_test))))

with open('RF_%s_TDDFT_energies.txt' % endpoint,'w') as refenergies:
    refenergies.write('%s\n' % (''.join(str(y_test))))

with open('RF_%s_train_TDDFT_energies.txt' % endpoint,'w') as refenergiestrain:
    refenergiestrain.write('%s\n' % (''.join(str(y_train))))

with open('RF_%s_predict.txt' % endpoint,'w') as predictfile:
    for el in prediction_test:
        predictfile.write('%s\n' % (''.join(el.astype(str))))

with open('RF_%s_trainpredict.txt' % endpoint,'w') as predictfiletrain:
    for el in prediction_train:
        predictfiletrain.write('%s\n' % (''.join(el.astype(str))))
