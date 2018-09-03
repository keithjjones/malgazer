# Machine Learning Module
import pandas as pd
import numpy as np
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.models import model_from_json
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, InputLayer
# import keras.callbacks
# from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import os
import dill
from .entropy import resample
from sklearn import tree
from fractions import Fraction
from collections import namedtuple


Keras_Dense_Parameters = namedtuple('Keras_Dense_Parameters', ['value', 'kernel_initializer', 'activation'])
Keras_Dropout_Parameters = namedtuple('Keras_Dropout_Parameters', 'rate')
Keras_Flatten_Parameters = namedtuple('Keras_Flatten_Parameters', [])
Keras_Conv1D_Parameters = namedtuple('Keras_Conv1D_Parameters', ['filters', 'kernel_size', 'activation', 'pool_size'])


class ML(object):
    def __init__(self, feature_type='rwe', classifier_type='dt', n_classes=6, *args, **kwargs):
        """
        A machine learning class to hold information about classifiers.

        :param feature_type:  Type of features for this ML package.
        """
        super(ML, self).__init__()
        self.classifer = None
        self.classifier_type = classifier_type
        if self.classifier_type:
            self.classifier_type = self.classifier_type.lower()
        self.n_classes = n_classes
        self.classifiers = None
        # X scaler
        self.X_sc = None
        # y label encoder
        self.y_labelencoder = None
        self.rwe_windowsize = kwargs.get('rwe_windowsize', None)
        self.datapoints = kwargs.get('datapoints', None)
        self.feature_type = feature_type

    def train(self, *args, **kwargs):
        if self.classifier_type == 'ann' or self.classifier_type == 'cnn':
            return self.train_nn(*args, **kwargs)
        else:
            return self.train_scikitlearn(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Perform a prediction on input data.

        :param args:  Passed through.
        :param kwargs:  Passed through.
        :return: The predictions.
        """
        if self.classifier_type == 'ann' or self.classifier_type == 'cnn':
            return self.predict_nn(*args, **kwargs)
        else:
            return self.predict_scikitlearn(*args, **kwargs)

    def predict_sample(self, sample, *args, **kwargs):
        """
        Perform a prediction on a single sample.

        :param sample:  The sample object to predict.
        :param args:  Passed through.
        :param kwargs:  Passed through.
        :return: The prediction.
        """
        if self.feature_type == 'rwe':
            ds1 = sample.running_window_entropy(self.rwe_windowsize)
            ds2 = pd.Series(resample(ds1, self.datapoints))
            ds2.name = ds1.name
            rwe = pd.DataFrame([ds2])
            rwe, _ = self.scale_features(rwe.values)
            y = self.decode_classifications(self.predict(rwe))
            return y[0]
        elif self.feature_type == 'gist':
            ds1 = sample.gist_data
            gist = pd.DataFrame([ds1])
            gist, _ = self.scale_features(gist.values)
            y = self.decode_classifications(self.predict(gist))
            return y[0]

    def save_classifier(self, directory):
        """
        Saves the classifier in directory with file name.

        :param directory:  Directory to save the classifier
        :param filename: Base file name of the classifier (without extensions)
        :return: Nothing
        """
        if self.classifier_type == 'gridsearch':
            raise TypeError("Gridsearch model saving is not supported!")
        if self.classifier_type == 'dt':
            tree.export_graphviz(self.classifier, out_file=os.path.join(directory, 'tree.dot'))
        if self.classifier_type in ['ann', 'cnn']:
            with open(os.path.join(directory, "nn.json"), 'w') as file:
                file.write(self.classifier.to_json())
            self.classifier.save_weights(os.path.join(directory, 'nn.h5'))
            self.classifier = None
            self.classifiers = None
        with open(os.path.join(directory, "ml.dill"), 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load_classifier(directory):
        """
        Load a classifier from JSON and H5 files.

        :param directory:  Directory containing the classifier.
        :param filename:  Base file name of the classifier (without extensions)
        :return:  The classifier
        """
        from keras.models import model_from_json
        with open(os.path.join(directory, "ml.dill"), 'rb') as file:
            myself = dill.load(file)
        if myself.classifier_type in ['ann', 'cnn']:
            with open(os.path.join(directory, "nn.json"), 'r') as file:
                myself.classifier = model_from_json(file.read())
            myself.classifier.load_weights(os.path.join(directory, 'nn.h5'))
        return myself

    def preprocess_data(self, X, y):
        """
        Pre processes data with label encoding and scaling.

        :param X:  The X input.
        :param y:  The y input.
        :return: A 2-tuple of X,y after encoding and scaling.
        """
        y_out, y_encoder = self.encode_classifications(y)
        X_out, X_scaler = self.scale_features(X)
        return X_out, y_out

    @staticmethod
    def build_gridsearch_static(*args, **kwargs):
        """
        Builds a Grid Search classifier.

        :return: The classifier.
        """
        classifier = GridSearchCV(*args, **kwargs)
        return classifier

    def build_gridsearch(self, gridsearch_type, *args, **kwargs):
        """
        Builds a Grid Search classifier.

        :return: The classifier.
        """
        self.classifier_type = 'gridsearch'
        self.base_classifier_type = gridsearch_type.lower()
        self.classifier = ML.build_gridsearch_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_ovr_static(*args, **kwargs):
        """
        Builds a OneVRest classifier.

        :return:  The classifier
        """
        classifier = OneVsRestClassifier(*args, **kwargs)
        return classifier

    def build_ovr(self, ovr_type, *args, **kwargs):
        """
        Builds a OneVRest classifier.

        :return:  The classifier
        """
        self.classifier_type = 'ovr'
        self.base_classifier_type = ovr_type.lower()
        self.classifier = ML.build_ovr_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_adaboost_static(*args, **kwargs):
        """
        Builds an AdaBoost classifier.

        :return:  The classifier
        """
        classifier = AdaBoostClassifier(*args, **kwargs)
        return classifier

    def build_adaboost(self, adaboost_type, *args, **kwargs):
        """
        Builds an AdaBoost classifier.

        :param adaboost_type:  The type of the base estimator.
        :return:  The classifier
        """
        self.classifier_type = 'adaboost'
        self.base_classifier_type = adaboost_type.lower()
        self.classifier = ML.build_adaboost_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_nc_static(*args, **kwargs):
        """
        Builds a Nearest Centroid classifier.

        :return:  The classifier
        """
        classifier = NearestCentroid(*args, **kwargs)
        return classifier

    def build_nc(self, *args, **kwargs):
        """
        Builds a Nearest Centroid classifier.

        :return:  The classifier
        """
        self.classifier_type = 'nc'
        self.classifier = ML.build_nc_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_nb_static(*args, **kwargs):
        """
        Builds a Naive Bayes classifier.

        :return:  The classifier
        """
        classifier = GaussianNB(*args, **kwargs)
        return classifier

    def build_nb(self, *args, **kwargs):
        """
        Builds a Naive Bayes classifier.

        :return:  The classifier
        """
        self.classifier_type = 'nb'
        self.classifier = ML.build_nb_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_knn_static(*args, **kwargs):
        """
        Builds a KNN classifier.

        :return:  The classifier
        """
        classifier = KNeighborsClassifier(*args, **kwargs)
        return classifier

    def build_knn(self, *args, **kwargs):
        """
        Builds a KNN classifier.

        :return:  The classifier
        """
        self.classifier_type = 'knn'
        self.classifier = ML.build_knn_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_rf_static(*args, **kwargs):
        """
        Builds a Random Forest classifier.

        :return:  The classifier
        """
        classifier = RandomForestClassifier(*args, **kwargs)
        return classifier

    def build_rf(self, *args, **kwargs):
        """
        Builds a Random Forest classifier.

        :return:  The classifier
        """
        self.classifier_type = 'rf'
        self.classifier = ML.build_rf_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_dt_static(*args, **kwargs):
        """
        Builds a Decision Tree classifier.

        :return:  The classifier
        """
        classifier = DecisionTreeClassifier(*args, **kwargs)
        return classifier

    def build_dt(self, *args, **kwargs):
        """
        Builds a Decision Tree classifier.

        :return:  The classifier
        """
        self.classifier_type = 'dt'
        self.classifier = ML.build_dt_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_svm_static(*args, **kwargs):
        """
        Builds an SVM classifier.

        :param kernel:  The SVM kernel to use.
        :return:  The classifier
        """
        classifier = SVC(*args, **kwargs)
        return classifier

    def build_svm(self, *args, **kwargs):
        """
        Builds an SVM classifier.

        :param kernel:  The SVM kernel to use.
        :return:  The classifier
        """
        self.classifier_type = 'svm'
        self.classifier = ML.build_svm_static(*args, **kwargs)
        return self.classifier

    def train_scikitlearn(self, X, y):
        """
        Trains a Scikit Learn classifier.

        :param X:  The X input
        :param y:  The y classifications
        :return:  The classifier
        """
        if self.classifier_type in ['svm', 'nb', 'nc', 'adaboost']:
            Y = y.argmax(1)
        elif self.classifier_type in ['gridsearch']:
            if self.base_classifier_type in ['svm', 'nb', 'nc', 'adaboost']:
                Y = y.argmax(1)
            else:
                Y = y
        else:
            Y = y
        self.classifier.fit(X, Y)
        return self.classifier

    def predict_scikitlearn(self, X):
        """
        Predict classifications using the classifier.

        :param X:  The data to predict.
        :return:  The predictions.
        """
        return self.classifier.predict(X)

    @staticmethod
    def build_ann_static(X, y, layers=[
        Keras_Dense_Parameters(value="1/2", kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value=100, kernel_initializer='uniform', activation='relu')
    ]):
        """
        Create a generic ANN.

        :param X:  The input to the ANN, used to find input shape.
        :param y:  The output to the ANN, used to find the output shape.
        :param layers:  A list of layer descriptions, where each item is a named tuple defined at the top of this
        file... where if the value is an int, the value will be used for the number of units for that layer, or
        if the value is a float or string it will be multiplied to the number of input data points.  kernel_initializer
        and activation are passed to the Dense object as such.
        :return:  The classifier.
        """
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        datapoints = X.shape[1]
        output_shape = y.shape[1]
        classifier = Sequential()
        classifier.add(Dense(units=datapoints, kernel_initializer='uniform',
                             activation='relu', input_dim=datapoints))

        for layer in layers:
            if isinstance(layer, Keras_Dense_Parameters):
                value, kernel_initializer, activation = layer.value, layer.kernel_initializer, layer.activation
                if isinstance(value, int):
                    units = value
                else:
                    units = int(Fraction(value) * datapoints)
                classifier.add(Dense(units=units,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            elif isinstance(layer, Keras_Dropout_Parameters):
                rate = layer.rate
                classifier.add(Dropout(rate))
            else:
                raise ValueError('Invalid layer type!')

        classifier.add(Dense(units=output_shape,
                             kernel_initializer='uniform',
                             activation='softmax'))
        classifier.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])
        return classifier

    def build_ann(self, X, y, layers=[("1/2", 'uniform', 'relu'), (100, 'uniform', 'relu')]):
        """
        Create a generic ANN.

        :param X:  The input to the ANN, used to find input shape.
        :param y:  The output to the ANN, used to find the output shape.
        :param layers:  A list of layer descriptions, where each item is a tuple:
            (value, kernel_initializer, activation)
        ... where if the value is an int, the value will be used for the number of units for that layer, or
        if the value is a float or string it will be multiplied to the number of input data points.  kernel_initializer
        and activation are passed to the Dense object as such.
        :return:  The classifier.
        """
        self.classifier_type = 'ann'
        self.classifier = ML.build_ann_static(X, y, layers=layers)
        self.classifier.summary()
        return self.classifier

    @staticmethod
    def build_cnn_static(X, y, layers=[
        Keras_Conv1D_Parameters(filters=10, kernel_size="1/4", activation="relu", pool_size=10),
        Keras_Conv1D_Parameters(filters=10, kernel_size="1/30", activation="relu", pool_size=2),
        Keras_Conv1D_Parameters(filters=10, kernel_size=2, activation='relu', pool_size=2),
        Keras_Flatten_Parameters(),
        Keras_Dense_Parameters(value="1/4", kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value="1/8", kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value="1/16", kernel_initializer='uniform', activation='relu')
    ]):
        """
        Create a generic CNN.

        :param X:  The input to the CNN, used to find input shape.
        :param y:  The output to the CNN, used to find the output shape.
        :param layers:  A list of layer descriptions for the CNN, where each item is a namedtuple from the top of
        this file... where if the value, filters, kernel_size, pool_size is an int, the corresponding value will be used, or
        if they are a a float or string they will be multiplied to the number of input data points.  kernel_initializer
        and activation are passed to the Dense and/or CNN object as such.  It makes more sense if you look at the short
        code below.
        :return:  The classifier.
        """
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, InputLayer
        datapoints = X.shape[1:]
        input_dim = datapoints[0]
        output_shape = y.shape[1]
        classifier = Sequential()
        classifier.add(InputLayer(input_shape=datapoints))

        print("input_dim: {0} datapoints {1}".format(input_dim, datapoints))

        for layer in layers:
            if isinstance(layer, Keras_Flatten_Parameters):
                classifier.add(Flatten())
            elif isinstance(layer, Keras_Conv1D_Parameters):
                filters, kernel_size, activation, pool_size = layer.filters, layer.kernel_size, \
                                                              layer.activation, layer.pool_size
                if not isinstance(filters, int):
                    filters = int(Fraction(filters) * input_dim)
                if not isinstance(kernel_size, int):
                    kernel_size = int(Fraction(kernel_size) * input_dim)
                if not isinstance(pool_size, int):
                    pool_size = int(Fraction(pool_size) * input_dim)

                classifier.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
                classifier.add(MaxPooling1D(pool_size=pool_size))
            elif isinstance(layer, Keras_Dense_Parameters):
                value, kernel_initializer, activation = layer.value, layer.kernel_initializer, layer.activation
                if isinstance(value, int):
                    units = value
                else:
                    units = int(Fraction(value) * input_dim)

                classifier.add(Dense(units=units,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            elif isinstance(layer, Keras_Dropout_Parameters):
                rate = layer.rate
                classifier.add(Dropout(rate))
            else:
                raise ValueError('Invalid layer type!')

        classifier.add(Dense(units=output_shape, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])
        return classifier

    def build_cnn(self, X, y):
        """
        Create a generic CNN.

        :param X:  The input to the CNN, used to find input shape.
        :param y:  The output to the CNN, used to find the output shape.
        :return:  The classifier.
        """
        if len(X.shape) != 3:
            X = np.expand_dims(X, axis=2)
        else:
            X = X
        self.classifier_type = 'cnn'
        self.classifier = ML.build_cnn_static(X, y)
        self.classifier.summary()
        return self.classifier

    def train_nn(self, X_train, y_train,
                 batch_size=50, epochs=100,
                 tensorboard=False):
        """
        Trains a given neural network with X_train and y_train.

        :param classifier:  The classifier to train.
        :param X_train:  The X training data.
        :param y_train:  The y training data.
        :param batch_size:  The batch size.
        :param epochs:  The number of epochs.
        :param tensorboard:  Set to True to include tensorboard
        data in the local directory under ./Graph
        :return: The classifier after training.
        """
        import keras.callbacks
        if len(X_train.shape) != 3 and self.classifier_type == 'cnn':
            X_in = np.expand_dims(X_train, axis=2)
        else:
            X_in = X_train
        if tensorboard is True:
            tb = keras.callbacks.TensorBoard(log_dir='Graph',
                                             histogram_freq=0,
                                             write_grads=True,
                                             write_graph=True,
                                             write_images=True)
            self.classifier.fit(X_in, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[tb])
        else:
            self.classifier.fit(X_in, y_train,
                                batch_size=batch_size,
                                epochs=epochs)
        return self.classifier

    def predict_nn(self, X_test):
        """
        Perform a prediction based upon our model.

        :param X_test:  The X testing data for the prediction.
        :return:  The predictions.
        """
        if len(X_test.shape) != 3 and self.classifier_type == 'cnn':
            X_in = np.expand_dims(X_test, axis=2)
        else:
            X_in = X_test
        y_pred = self.classifier.predict(X_in)
        # Pick the best match
        for i in range(0, len(y_pred)):
            row = y_pred[i]
            row[row == row.max()] = 1
            row[row < row.max()] = 0
            y_pred[i] = row
        return y_pred

    @staticmethod
    def confusion_matrix(y_test, y_pred):
        """
        Calculates the confusion matrix.

        :param y_test:  The y testing data.
        :param y_pred:  The y predicted data.
        :return:  The accuracy,confusion_matrix, as a tuple.
        """
        if isinstance(y_test, list):
            y_test = np.array(y_test)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            yp = column_or_1d(y_pred.argmax(1)).tolist()
        else:
            yp = column_or_1d(y_pred).tolist()
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            yt = column_or_1d(y_test.argmax(1)).tolist()
        else:
            yt = column_or_1d(y_test).tolist()
        cm = confusion_matrix(yt, yp)
        return ML._calculate_confusion_matrix(cm)

    @staticmethod
    def _calculate_confusion_matrix(cm):
        """
        Internal method to calculate statistics from a confusion matrix.

        :param cm:  A confusion matrix from scikit learn
        :return: The accuracy, confusion_matrix, as a tuple
        """
        accuracy = 0.
        for i in range(0, len(cm)):
            accuracy += cm[i, i]
        accuracy = accuracy/cm.sum()
        return accuracy, cm

    def scale_features(self, X):
        """
        Scales features in the X data.

        :param X:  The data to scale.
        :return: A tuple of X_scaled and the scaler as X_scaled, scaler
        """
        if self.X_sc is None:
            self.X_sc = StandardScaler()
            X_scaled = self.X_sc.fit_transform(X)
        else:
            X_scaled = self.X_sc.transform(X)
        # onehotencoder = OneHotEncoder(categorical_features = [0])
        # y = onehotencoder.fit_transform(y).toarray()
        # inverted = labelencoder_y.inverse_transform([argmax(y)])
        # y = y[:, 1:]
        return X_scaled, self.X_sc

    def encode_classifications(self, y):
        """
        Encodes the classifications.

        :param y:  The preprocessed data as a DataFrame.
        :return:  A tuple of the encoded data y and the encoder (for inverting)
        as y,encoder.
        """
        from keras.utils import to_categorical
        if self.y_labelencoder is None:
            self.y_labelencoder = LabelEncoder()
            y[:, 0] = self.y_labelencoder.fit_transform(y[:, 0])
        else:
            y[:, 0] = self.y_labelencoder.transform(y[:, 0])
        y = to_categorical(y)
        return y, self.y_labelencoder

    def decode_classifications(self, y):
        """
        Decodes the classifications.

        :param y:  The preprocessed data as a DataFrame.
        :return:  The decoded data y.
        """
        if self.y_labelencoder is not None:
            y = np.argmax(y, axis=1)
            y_out = self.y_labelencoder.inverse_transform(y)
            return y_out
        else:
            return None

    @staticmethod
    def train_test_split(X, y, test_percent=0.2, random_state=0):
        """
        Creates a training and testing data sets.

        :param X:  The X values as a DataFrame.
        :param y:  The y values as a DataFrame.
        :param test_percent: The percentage, as a decimal, of the test data set size.
        :param random_state:  The random seed.
        :return: A tuple of X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_percent,
                                                            random_state=random_state,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test

    def cross_fold_validation(self, X, y, cv=10, n_jobs=10, batch_size=None, epochs=None):
        """
        Calculates the cross fold validation mean and variance of models.

        :param X:  The X training data.
        :param y:  The y training data.
        :param cv:  The number of cfv groups.
        :param n_jobs:  The number of jobs to run at once.
        :param batch_size:  The batch size for Keras classifiers.
        :param epochs:  The number of epochs for Keras classifiers.
        :return:  A tuple of mean, variance, classifiers (dict).
        """
        cvkfold = StratifiedKFold(n_splits=cv)

        Y = y.argmax(1)

        fold = 0
        saved_futures = {}
        classifiers = {}
        print("Start Cross Fold Validation...")

        if n_jobs < 2:
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=n_jobs)

        for train, test in cvkfold.split(X, Y):
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]
            fold += 1
            print("\tCalculating fold: {0}".format(fold))
            if executor:
                future = executor.submit(self._cfv_runner,
                                     X_train, y_train,
                                     X_test, y_test,
                                     batch_size=batch_size, epochs=epochs)
                saved_futures[future] = fold
                if len(saved_futures) >= n_jobs:
                    keystodel = []
                    futures = wait(saved_futures, return_when=FIRST_COMPLETED)
                    for future in futures.done:
                        print("\tFinished calculating fold: {0}".format(saved_futures[future]))
                        result_dict = future.result()
                        classifiers[saved_futures[future]] = result_dict
                        keystodel.append(future)
                    for key in keystodel:
                        del saved_futures[key]
            else:
                result_dict = self._cfv_runner(X_train, y_train, X_test, y_test, batch_size=batch_size, epochs=epochs)
                print("\tFinished calculating fold: {0}".format(fold))
                classifiers[fold] = result_dict

        if executor:
            for future in as_completed(saved_futures):
                print("\tFinished calculating fold: {0}".format(saved_futures[future]))
                result_dict = future.result()
                classifiers[saved_futures[future]] = result_dict
            executor.shutdown(wait=True)

        self.classifiers = classifiers
        accuracies = np.array([classifiers[f]['accuracy'] for f in classifiers])
        mean = accuracies.mean()
        variance = accuracies.std()
        return mean, variance, classifiers

    def _cfv_runner(self, X_train, y_train, X_test, y_test, batch_size=None, epochs=None, **kwargs):
        """
        Internal method for multi-processing to calculate the CFV of a model.

        :param X_train:  The X training set.
        :param y_train:  The Y training set.
        :param X_test:  The X testing set.
        :param y_test:  The X testing set.
        :param batch_size:  The batch size for Keras classifiers.
        :param epochs:  The number of epochs for Keras classifiers.
        :return:  A dictionary with the results.
        """
        if self.classifier_type in ['ann', 'cnn']:
            from keras.wrappers.scikit_learn import KerasClassifier
            if self.classifier_type == 'cnn':
                X_train_in = np.expand_dims(X_train, axis=2)
            else:
                X_train_in = X_train

            def create_model():
                if self.classifier_type == 'cnn':
                    return ML.build_cnn_static(X_train_in, label_binarize(y_train, classes=range(self.n_classes)))
                else:
                    return ML.build_ann_static(X_train, label_binarize(y_train, classes=range(self.n_classes)))
            classifier = KerasClassifier(build_fn=create_model,
                                         batch_size=batch_size,
                                         epochs=epochs)
            classifier.fit(X_train_in, label_binarize(y_train, classes=range(self.n_classes)), batch_size=batch_size, epochs=epochs, **kwargs)
        else:
            classifier = self.classifier
            if self.classifier_type in ['svm', 'nb', 'nc', 'adaboost']:
                Y = y_train.argmax(1)
            else:
                Y = y_train
            classifier.fit(X_train, Y, **kwargs)
        # probas = classifier_type.predict_proba(X_test)
        if self.classifier_type == 'cnn':
            X_test_in = np.expand_dims(X_test, axis=2)
            y_pred = classifier.predict(X_test_in, **kwargs)
        else:
            y_pred = classifier.predict(X_test, **kwargs)
        accuracy, cm = ML.confusion_matrix(y_test, y_pred)
        return_dict = {'classifier': classifier, 'cm': cm, 'accuracy': accuracy,
                       'y_test': np.array(y_test),
                       # 'y_train': np.array(y_train),
                       'y_pred': np.array(y_pred),
                       # 'X_test': np.array(X_test), 'X_train': np.array(X_train),
                       'type': self.classifier_type}
        if self.classifier_type in ['ann', 'cnn']:
            classifier_dict = {}
            classifier_dict['json'] = classifier.model.to_json()
            classifier_dict['weights'] = classifier.model.get_weights()
            return_dict['classifier'] = classifier_dict
        return return_dict

    def set_classifier_by_fold(self, fold):
        """
        Sets the classifier.  This is useful after picking the best cross fold
        validated classifier, for example.

        :param fold:  The classifier fold number.
        :return: Nothing.
        """
        from keras.models import model_from_json
        if self.classifiers:
            if self.classifiers[fold]['type'] == 'keras':
                self.classifier = model_from_json(self.classifiers[fold]['classifier']['json'])
                self.classifier.set_weights(self.classifiers[fold]['classifier']['weights'])
            else:
                self.classifier = self.classifiers[fold]['classifier']
        else:
            raise AttributeError("Must use CFV before there are classifiers to set.")

    def plot_roc_curves(self, y_test, y_pred, fold=None, filename=None):
        """
        Plot ROC curves for the data and classifier.

        :param y_test:  The y testing data.
        :param y_pred:  The y predicted data.
        :param fold:  An optional fold number to add to the title.
        :param filename:  A PNG filename to save this to.  Set to None to not save to a file.
        :return: Nothing.  This plots the curve.
        """
        if isinstance(y_test, list):
            y_test = np.array(y_test)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        # Compute micro-average ROC curve and ROC area
        yt = y_test
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            yp = y_pred
        else:
            yp = label_binarize(y_pred.tolist(), classes=range(self.n_classes))
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        for i in self.n_classes:
            fpr[i], tpr[i], thresholds[i] = roc_curve(yt[:, i], yp[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(yt.ravel(), yp.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        lw = 2
        for i, color in zip(range(self.n_classes),
                            ['aqua', 'darkorange', 'cornflowerblue', 'red',
                             'green', 'yellow']):
            cn = label_binarize([i], classes=range(self.n_classes))
            class_name = self.decode_classifications(cn)[0]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve for class {0} (area = {1:0.2f})'.format(class_name, roc_auc[i]))
        plt.plot(fpr["micro"], tpr["micro"], color='darkmagenta',
                 lw=lw, label='Micro ROC curve (area = {0:2f})'.format(roc_auc["micro"]))
        plt.plot(fpr["macro"], tpr["macro"], color='darkorange',
                 lw=lw, label='Macro ROC curve (area = {0:2f})'.format(roc_auc["macro"]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if fold:
            plt.title('Receiver operating characteristic Fold={0}'.format(fold))
        else:
            plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        try:
            plt.show()
        except Exception as exc:
            print("UNHANDLED EXCEPTION - Trying to show Matplotlib plot: {0}".format(exc))
            raise

        # Save the figure...
        if filename:
            plt.savefig(filename)
