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
datapoints = 2048
datadir = os.path.join('/Volumes/MALWARE 1/Focused Set May 2018', 'data_vt_{0}'.format(datapoints))
arguments = ['-w', '256', '-d', str(datapoints), '-j', '500', source_dir, datadir]
batch_size = 100
epochs = 100

#classifications = Utils.get_classifications_from_path(source_dir)
#classifications = Utils.estimate_vt_classifications_from_csv('/Dirty/Samples/all_vt_data.csv')
#classifications.to_csv(os.path.join(datadir, 'classifications.csv'))

# Uncomment to process data
batch_preprocess_entropy.main(arguments)

# Load data
raw_data_tmp, classifications_tmp = Utils.load_preprocessed_data(datadir)
all_data, raw_data, classifications = Utils.sanity_check_classifications(raw_data_tmp, classifications_tmp)
#X = raw_data.as_matrix().copy()
#y = classifications.as_matrix().copy()
X = all_data.drop('classification', axis=1).as_matrix().copy()
y = pd.DataFrame(all_data['classification']).as_matrix().copy()


# Preprocess the data
ml = ML()
y, y_encoder = ml.encode_classifications(y)
X, X_scaler = ml.scale_features(X)
X_train, X_test, y_train, y_test = ml.train_test_split(X, y)
Xt = np.expand_dims(X_train, axis=2)
yt = y_train
outputs = yt.shape[1]

# Create the CNN
classifier = ml.build_cnn(Xt, outputs)

# Train the CNN
start_time = time.time()
classifier = ml.train_nn(Xt, yt, batch_size=batch_size, epochs=epochs)
print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))

# Predict the results
Xtest = np.expand_dims(X_test, axis=2)
y_pred = ml.predict_nn(Xtest)

# Making the Confusion Matrix
accuracy, cm = ml.confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print(accuracy)

## Cross Fold Validation
#def model():
#    return ML.build_cnn_static(Xt, outputs)
#accuracies, mean, variance = ml.cross_fold_validation(model,
#                                                      Xt, yt, 
#                                                      batch_size=batch_size, 
#                                                      epochs=epochs, 
#                                                      cv=10, 
#                                                      n_jobs=2)
#print("CFV Mean: {0}".format(mean))
#print("CFV Var: {0}".format(variance))

