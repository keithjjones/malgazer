import batch_preprocess_entropy
from library.utils import Utils
from library.ml import ML
import pandas as pd
import numpy as np

# Calculate features
source_dir = '/Dirty/malgazer/Test_Set/'
datapoints = 1024
subdir = 'data_{0}'.format(datapoints)
arguments = ['-w', '256', '-d', str(datapoints), source_dir, subdir]
batch_size = 10
epochs = 100

#classifications = Utils.get_classifications_from_path(source_dir)

# Uncomment to process data
#batch_preprocess_entropy.main(arguments)

# Load data
raw_data, classifications = Utils.load_preprocessed_data(subdir)

# Wrangle classifications
cls = Utils.parse_classifications_from_path(classifications)
X = raw_data.as_matrix().copy()
y = cls.as_matrix().copy()

# Preprocess the data
ml = ML()
y, y_encoder = ml.encode_preprocessed_data(y)
X, X_scaler = ml.scale_features_preprocessed_data(X)
X_train, X_test, y_train, y_test = ml.train_test_split(X, y)
Xt = np.expand_dims(X_train, axis=2)
yt = y_train
outputs = yt.shape[1]

# Create the CNN
classifier = ml.build_cnn(Xt, outputs)

# Train the CNN
classifier = ml.train_nn(Xt, yt, batch_size=batch_size, epochs=epochs)

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

