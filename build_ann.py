# Install the plaidml backend
#import plaidml.keras
#plaidml.keras.install_backend()

import batch_preprocess_entropy
from library.utils import Utils
from library.ml import ML
import pandas as pd
import numpy as np
import time
import os

pd.set_option('max_colwidth', 64)

# Calculate features
source_dir = '/Volumes/MALWARE 1/Focused Set May 2018/RWE'
datapoints = 1024
datadir = os.path.join('/Volumes/MALWARE 1/Focused Set May 2018', 'data_vt_{0}'.format(datapoints))
arguments = ['-w', '256', '-d', str(datapoints), '-j', '50', source_dir, datadir]
batch_size = 100
epochs = 100

#classifications = Utils.get_classifications_from_path(source_dir)
#classifications = Utils.estimate_vt_classifications_from_csv('/Dirty/Samples/all_vt_data.csv')
#classifications.to_csv(os.path.join(datadir, 'classifications.csv'))

# Uncomment to process data
#batch_preprocess_entropy.main(arguments)

# Load data
raw_data_tmp, classifications_tmp = Utils.load_preprocessed_data(datadir)
all_data, raw_data, classifications = Utils.sanity_check_classifications(raw_data_tmp, classifications_tmp)
#X = raw_data.as_matrix().copy()
#y = classifications.as_matrix().copy()
X = all_data.drop('classification', axis=1).values
y = all_data['classification'].values


# Preprocess the data
ml = ML()
y, y_encoder = ml.encode_preprocessed_data(y)
X, X_scaler = ml.scale_features_preprocessed_data(X)
X_train, X_test, y_train, y_test = ml.train_test_split(X, y)
outputs = y_train.shape[1]

# Create the ANN
classifier = ml.build_ann(datapoints, outputs)

# Train the ANN
start_time = time.time()
classifier = ml.train_nn(X_train, y_train, batch_size=batch_size, epochs=epochs, tensorboard=False)
print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))

# Predict the results
y_pred = ml.predict_nn(X_test)

# Making the Confusion Matrix
accuracy, cm = ml.confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print(accuracy)

# Cross Fold Validation
#def model():
#    return ML.build_ann_static(datapoints, outputs)
#accuracies, mean, variance = ML.cross_fold_validation(model,
#                                                      X_train, y_train, 
#                                                      batch_size=batch_size, 
#                                                      epochs=epochs, 
#                                                      cv=10, 
#                                                      n_jobs=2)
#print("CFV Mean: {0}".format(mean))
#print("CFV Var: {0}".format(variance))
