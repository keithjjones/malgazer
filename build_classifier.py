# Install the plaidml backend
#import plaidml.keras
#plaidml.keras.install_backend()

import batch_preprocess_entropy
from library.utils import Utils
from library.ml import ML
from sklearn.utils.validation import column_or_1d
import pandas as pd
import numpy as np
import time
import os
import pickle

pd.set_option('max_colwidth', 64)

#
# Script actions
#
# Preprocess the data into one DataFrame
preprocess_data = False
# Assemble the preprocessed data
assemble_preprocessed_data = False
# Build a classifier
build_classifier = True
classifier_type = 'svm'

#
# Calculate features
#
source_dir = '/Volumes/JONES/Focused Set May 2018/RWE'
datapoints = 1024
windowsize = 256
number_of_jobs = 50
datadir = os.path.join('/Volumes/JONES/Focused Set May 2018', 'data_vt_window_{0}_samples_{1}'.format(windowsize, datapoints))
arguments = ['-w', str(windowsize), '-d', str(datapoints), '-j', str(number_of_jobs), source_dir, datadir]
batch_size = 100
epochs = 10
test_size = 0.5
class_size = 1000

categorical = True
if classifier_type.lower() == 'svm':
    categorical = False

# Preprocess data
if preprocess_data:
    batch_preprocess_entropy.main(arguments)

# Put the data together and save hashes used for training
if assemble_preprocessed_data:
    # Load data
    raw_data_tmp, classifications_tmp = Utils.load_preprocessed_data(datadir)
    
    # Make sure data lines up
    all_data, raw_data, classifications = Utils.sanity_check_classifications(raw_data_tmp, classifications_tmp)
    
    # Pick 60k samples, 10k from each classification
    trimmed_data = all_data.groupby('classification').head(10000)
    trimmed_data.to_csv(os.path.join(datadir, 'data.csv'))
    pd.DataFrame(trimmed_data.index).to_csv(os.path.join(datadir, 'hashes_60k.txt'), header=False, index=False)
    
    # Pull the hashes we care about
    hashes = pd.read_csv(os.path.join(datadir, 'hashes_60k.txt'), header=None).values[:,0]
    data = Utils.filter_hashes(all_data, hashes)
    data.to_csv(os.path.join(datadir, 'data.csv'))

# Build a classifier
if build_classifier:
    # Load data
    raw_data_tmp, classifications_tmp = Utils.load_preprocessed_data(datadir)
    
    # Make sure data lines up
    all_data, raw_data, classifications = Utils.sanity_check_classifications(raw_data_tmp, classifications_tmp)
    
    # Pull the hashes we care about
    hashes = pd.read_csv(os.path.join(datadir, 'hashes_60k.txt'), header=None).values[:,0]
    data = Utils.filter_hashes(all_data, hashes)
    data.to_csv(os.path.join(datadir, 'data.csv'))
    
    # Reduce the data set for experimentation if class_size is not None
    if class_size:
        data = data.groupby('classification').head(class_size)
        print(data['classification'].value_counts())
    
    # Read in the final training data
    #data = pd.read_csv(os.path.join(datadir, 'data.csv'), index_col=0)
    X = data.drop('classification', axis=1).as_matrix().copy()
    y = pd.DataFrame(data['classification']).as_matrix().copy()
    
    # Make the classifier
    ml = ML()
    y, y_encoder = ml.encode_classifications(y, categorical=categorical)
    X, X_scaler = ml.scale_features(X)
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_size=test_size)

    if classifier_type.lower() == 'cnn':    
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
        accuracy, cm = ml.confusion_matrix_nn(y_test, y_pred)
        
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
    elif classifier_type.lower() == 'ann':
        outputs = y_train.shape[1]
        # Create the ANN
        classifier = ml.build_ann(datapoints, outputs)
        # Train the NN
        start_time = time.time()
        classifier = ml.train_nn(X_train, y_train, batch_size=batch_size, epochs=epochs, tensorboard=False)
        print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))                
        # Predict the results
        Xtest = np.expand_dims(X_test, axis=2)
        y_pred = ml.predict_nn(X_test)
        
        # Making the Confusion Matrix
        accuracy, cm = ml.confusion_matrix_nn(y_test, y_pred)
        
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
    elif classifier_type.lower() == 'svm':
        classifier = ml.build_svm()
        start_time = time.time()
        classifier = ml.train_svm(X_train, y_train)
        print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
        y_pred = ml.predict_svm(X_test)

        # Making the Confusion Matrix
        accuracy, cm = ml.confusion_matrix_standard(y_test, y_pred)

        print("Confusion Matrix:")
        print(cm)
        print("Accuracy:")
        print(accuracy)

    # Save the classifier
    print("Saving the classifier...")
    path = os.path.join(datadir, classifier_type.lower())
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    ml.save_classifier(path, "classifier")
