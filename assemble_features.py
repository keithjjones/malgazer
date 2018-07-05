# Install the plaidml backend
#import plaidml.keras
#plaidml.keras.install_backend()

from library.utils import Utils
from library.ml import ML, column_or_1d
from sklearn.utils.validation import column_or_1d
import pandas as pd
import numpy as np
import time
import os
import pickle
import dill
from sklearn import tree
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import label_binarize


pd.set_option('max_colwidth', 64)

#
# Script actions
#
feature_type = 'rwe'

#
# Calculate features
#
source_dir = '/Volumes/JONES/Focused Set May 2018/Binaries'

datapoints = 1024
windowsize = 256
n_categories = 6
if feature_type == 'rwe':
    datadir = '/Volumes/JONES/Focused Set May 2018/RWE'
elif feature_type == 'gist':
    datadir = '/Volumes/JONES/Focused Set May 2018/GIST'

# Load data
all_data, raw_data, classifications = Utils.load_features(datadir, feature_type, windowsize=windowsize, datapoints=datapoints)

# Pick 60k samples, 10k from each classification
trimmed_data = all_data.groupby('classification').head(10000)
# trimmed_data.to_csv(os.path.join(datadir, 'data.csv'))
pd.DataFrame(trimmed_data.index).to_csv(os.path.join(datadir, 'hashes_60k.txt'), header=False, index=False)

# Pull the hashes we care about
hashes = pd.read_csv(os.path.join(datadir, 'hashes_60k.txt'), header=None).values[:,0]
data = Utils.filter_hashes(all_data, hashes)
data.to_csv(os.path.join(datadir, 'data.csv'))