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
from sklearn import tree


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
classifier_type = 'nc'

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

# Cross fold validation variables
cross_fold_validation = False
cfv_groups = 6
n_jobs = 10

# KNN params
knn_neighbors = 20

# Centroid params
shrink_threshold = 0.2

# Set this to the percentage for test size, 
# 0 makes the train and test set be the whole data set
test_percent = 0.5
# Make the whole data set for training if we are doing cross fold validation
if cross_fold_validation is True:
    test_percent = 0

# We don't need categorial translation for some models
categorical = False
if classifier_type.lower() in ['ann', 'cnn']:
    categorical = True

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
    print("Loading data...")
    
    # Load data
    raw_data_tmp, classifications_tmp = Utils.load_preprocessed_data(datadir)
    
    # Make sure data lines up
    all_data, raw_data, classifications = Utils.sanity_check_classifications(raw_data_tmp, classifications_tmp)
    
    # Pull the hashes we care about
    hashes = pd.read_csv(os.path.join(datadir, 'hashes_60k.txt'), header=None).values[:,0]
    data = Utils.filter_hashes(all_data, hashes)
    data.to_csv(os.path.join(datadir, 'data.csv'))
        
    print("Test percent: {0}".format(test_percent))
    
    # Read in the final training data
    #data = pd.read_csv(os.path.join(datadir, 'data.csv'), index_col=0)
    X = data.drop('classification', axis=1).as_matrix().copy()
    y = pd.DataFrame(data['classification']).as_matrix().copy()
    
    # Make the classifier
    ml = ML()
    y, y_encoder = ml.encode_classifications(y, categorical=categorical)
    X, X_scaler = ml.scale_features(X)
    if test_percent > 0:
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_percent=test_percent)
    else:
        X_train = X
        y_train = y
        X_test = X
        y_test = y

    print("Training Class Count: \n{0}".format(pd.DataFrame(y_train)[0].value_counts()))
    print("Testing Class Count: \n{0}".format(pd.DataFrame(y_test)[0].value_counts()))

    print("Beginning training...")

    if classifier_type.lower() == 'cnn':    
        Xt = np.expand_dims(X_train, axis=2)
        yt = y_train
        outputs = yt.shape[1]
        if cross_fold_validation is False:
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
        else:        
            # Cross Fold Validation
            def model():
                return ML.build_cnn_static(Xt, outputs)
            start_time = time.time()
            accuracies, mean, variance = ml.cross_fold_validation_keras(model,
                                                                        Xt, yt, 
                                                                        batch_size=batch_size, 
                                                                        epochs=epochs, 
                                                                        cv=cfv_groups, 
                                                                        n_jobs=n_jobs)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print("CFV Mean: {0}".format(mean))
            print("CFV Var: {0}".format(variance))
    elif classifier_type.lower() == 'ann':
        outputs = y_train.shape[1]
        if cross_fold_validation is False:        
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
        else:
            # Cross Fold Validation
            def model():
                return ML.build_ann_static(datapoints, outputs)
            start_time = time.time()
            accuracies, mean, variance = ML.cross_fold_validation_keras(model,
                                                                        X_train, y_train, 
                                                                        batch_size=batch_size, 
                                                                        epochs=epochs, 
                                                                        cv=cfv_groups, 
                                                                        n_jobs=n_jobs)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print("CFV Mean: {0}".format(mean))
            print("CFV Var: {0}".format(variance))
    else:
        # This area is for scikit learn models
        if classifier_type.lower() == 'svm':
            classifier = ml.build_svm(kernel='rbf')
        elif classifier_type.lower() == 'dt':
            classifier = ml.build_dt(criterion='entropy')
        elif classifier_type.lower() == 'nb':
            classifier = ml.build_nb()
        elif classifier_type.lower() == 'rf':
            classifier = ml.build_rf(n_estimators=10, criterion='entropy')
        elif classifier_type.lower() == 'knn':
            classifier = ml.build_knn(n_neighbors=knn_neighbors, n_jobs=n_jobs)
        elif classifier_type.lower() == 'nc':
            classifier = ml.build_nc(shrink_threshold=shrink_threshold)
            
        start_time = time.time()
        if cross_fold_validation is False:
            classifier = ml.train_scikitlearn(X_train, y_train)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            y_pred = ml.predict_scikitlearn(X_test)
    
            # Making the Confusion Matrix
            accuracy, cm = ml.confusion_matrix_scikitlearn(y_test, y_pred)
    
            print("Confusion Matrix:")
            print(cm)
            print("Accuracy:")
            print(accuracy)
        else:
            # Cross Fold Validation
            accuracies, mean, variance = ML.cross_fold_validation_scikitlearn(classifier,
                                                                              X_train, y_train, 
                                                                              cv=cfv_groups, 
                                                                              n_jobs=n_jobs)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print("CFV Mean: {0}".format(mean))
            print("CFV Var: {0}".format(variance))

    # Save the classifier
    if cross_fold_validation is False:
        print("Saving the classifier...")
        path = os.path.join(datadir, classifier_type.lower())
        try:
            os.stat(path)
        except:
            os.mkdir(path)
        ml.save_classifier(path, "classifier")
        if classifier_type.lower() == 'dt':
            tree.export_graphviz(classifier, out_file=os.path.join(path, 'tree.dot'))
