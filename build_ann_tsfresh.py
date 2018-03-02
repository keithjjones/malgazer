import batch_tsfresh_entropy
from library.utils import Utils
from library.ml import ML
import pandas as pd
import numpy as np

# Calculate features
source_dir = '/Dirty/malgazer/Test_Set/'
datapoints = 1024
subdir = 'data_tsfresh_{0}'.format(datapoints)
arguments = ['-w', '256', '-d', str(datapoints), source_dir, subdir]
outputs = 9

#classifications = Utils.get_classifications_from_path(source_dir)

# Uncomment to process data
#batch_tsfresh_entropy.main(arguments)

# Load data
raw_data, classifications, extracted_features = Utils.load_processed_tsfresh_data(subdir)

# Wrangle classifications
cls = Utils.parse_classifications_from_path(classifications)
X = extracted_features.as_matrix().copy()
y = cls.as_matrix().copy()

# Preprocess the data
ml = ML()
y, y_encoder = ml.encode_preprocessed_data(y)
X, X_scaler = ml.scale_features_preprocessed_data(X)
X_train, X_test, y_train, y_test = ml.train_test_split(X, y)

# Create the ANN
classifier = ml.build_ann(788, outputs)

# Train the ANN
classifier = ml.train_nn(X_train, y_train, batch_size=10, epochs=100)

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
#    return ML.build_ann_static(788, outputs)
#accuracies, mean, variance = ml.cross_fold_validation(model,
#                                                      X_train, y_train, 
#                                                      batch_size=10, 
#                                                      epochs=100, 
#                                                      cv=10, 
#                                                      n_jobs=2)
#print("CFV Mean: {0}".format(mean))
#print("CFV Var: {0}".format(variance))
