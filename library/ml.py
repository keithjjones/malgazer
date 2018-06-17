# Machine Learning Module
import pandas as pd
import numpy as np
from numpy import argmax
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, InputLayer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
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


import keras.backend as K
import keras.callbacks
import os
import pickle


class ML(object):
    def __init__(self):
        super(ML, self).__init__()
        self.classifer = None
        self.classifier_type = None
        # X scaler
        self.X_sc = None
        # y label encoder
        self.y_labelencoder = None

    def save_classifier(self, directory, filename):
        """
        Saves the classifier in directory with file name.

        :param directory:  Directory to save the classifier
        :param filename: Base file name of the classifier (without extensions)
        :return: Nothing
        """
        if self.classifer and (self.classifier_type == 'ann' or self.classifier_type == 'cnn'):
            with open(os.path.join(directory, filename+".json"), 'w') as file:
                file.write(self.classifier.to_json())
            self.classifier.save_weights(os.path.join(directory, filename+'.h5'))
        else:
            with open(os.path.join(directory, filename+".pickle"), 'wb') as file:
                pickle.dump(self.classifier, file)

    def load_classifier(self, directory, filename, classifier_type):
        """
        Load a classifier from JSON and H5 files.

        :param directory:  Directory containing the classifier.
        :param filename:  Base file name of the classifier (without extensions)
        :return:  The classifier
        """
        if self.classifier_type == 'ann' or self.classifier_type == 'cnn':
            with open(os.path.join(directory, filename + ".json"), 'r') as file:
                self.classifer = model_from_json(file.read())
            self.classifer.load_weights(os.path.join(directory, filename + '.h5'))
            return self.classifer
        else:
            with open(os.path.join(directory, filename+".pickle"), 'rb') as file:
                return pickle.load(file)

    @staticmethod
    def build_ovr_static(*args, **kwargs):
        """
        Builds a OneVRest classifier.

        :return:  The classifier
        """
        classifier = OneVsRestClassifier(*args, **kwargs)
        return classifier

    def build_ovr(self, *args, **kwargs):
        """
        Builds a OneVRest classifier.

        :return:  The classifier
        """
        self.classifier_type = 'ovr'
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

    def build_adaboost(self, *args, **kwargs):
        """
        Builds an AdaBoost classifier.

        :return:  The classifier
        """
        self.classifier_type = 'adaboost'
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
        self.classifier.fit(X, column_or_1d(y).tolist())
        return self.classifier

    def predict_scikitlearn(self, X):
        """
        Predict classifications using the classifier.

        :param X:  The data to predict.
        :return:  The predictions.
        """
        return self.classifier.predict(X)

    @staticmethod
    def build_ann_static(datapoints=1024, outputs=9):
        """
        Create a generic ANN.

        :param datapoints:  The number of data points on the input.
        :return:  The classifier.
        """
        classifier = Sequential()
        classifier.add(Dense(units=datapoints, kernel_initializer='uniform',
                             activation='relu', input_dim=datapoints))
        classifier.add(Dense(units=int(datapoints / 2),
                             kernel_initializer='uniform',
                             activation='relu'))
        classifier.add(Dense(units=100,
                             kernel_initializer='uniform',
                             activation='relu'))
        classifier.add(Dense(units=outputs,
                             kernel_initializer='uniform',
                             activation='softmax'))
        classifier.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])
        return classifier

    def build_ann(self, datapoints=1024, outputs=9):
        """
        Create a generic ANN.

        :param datapoints:  The number of data points on the input.
        :return:  The classifier.
        """
        self.classifier_type = 'ann'
        self.classifier = ML.build_ann_static(datapoints, outputs)
        self.classifier.summary()
        return self.classifier

    @staticmethod
    def build_cnn_static(input, outputs=9):
        """
        Create a generic CNN.

        :param input:  The input to the CNN, used to find input shape.
        :return:  The classifier.
        """
        classifier = Sequential()
        classifier.add(InputLayer(input_shape=input.shape[1:]))
        classifier.add(Conv1D(filters=100, kernel_size=100, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=16))
        classifier.add(Conv1D(filters=100, kernel_size=20, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=4))
        classifier.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=2))
        classifier.add(Flatten())
        classifier.add(Dense(units=700, activation='relu'))
        classifier.add(Dense(units=250, activation='relu'))
        classifier.add(Dense(units=50, activation='relu'))
        classifier.add(Dense(units=outputs, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])
        return classifier

    def build_cnn(self, input, outputs=9):
        """
        Create a generic CNN.

        :param input:  The input to the CNN, used to find input shape.
        :return:  The classifier.
        """
        self.classifier_type = 'cnn'
        self.classifier = ML.build_cnn_static(input, outputs)
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
        if tensorboard is True:
            tb = keras.callbacks.TensorBoard(log_dir='Graph',
                                             histogram_freq=0,
                                             write_grads=True,
                                             write_graph=True,
                                             write_images=True)
            self.classifier.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[tb])
        else:
            self.classifier.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs)

        return self.classifier

    def predict_nn(self, X_test):
        """
        Perform a prediction based upon our model.

        :param X_test:  The X testing data for the prediction.
        :return:  The predictions.
        """
        y_pred = self.classifier.predict(X_test)
        # Pick the best match
        for i in range(0, len(y_pred)):
            row = y_pred[i]
            row[row == row.max()] = 1
            row[row < row.max()] = 0
            y_pred[i] = row
        return y_pred

    @staticmethod
    def confusion_matrix_nn(y_test, y_pred):
        """
        Calculates the confusion matrix for neural network predictions.

        :param y_test:  The y testing data.
        :param y_pred:  The y predicted data.
        :return:  The accuracy,confusion_matrix, as a tuple.
        """
        cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
        return ML._calculate_confusion_matrix(cm)

    @staticmethod
    def confusion_matrix_scikitlearn(y_test, y_pred):
        """
        Calculates the standard confusion matrix for predictions.

        :param y_test:  The y testing data.
        :param y_pred:  The y predicted data.
        :return:  The accuracy,confusion_matrix, as a tuple.
        """
        cm = confusion_matrix(column_or_1d(y_test).tolist(), column_or_1d(y_pred).tolist())
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

    def encode_classifications(self, y, categorical=True):
        """
        Encodes the classifications.

        :param y:  The preprocessed data as a DataFrame.
        :param categorical:  Set to true to run to_categorical.
        :return:  A tuple of the encoded data y and the encoder (for inverting)
        as y,encoder.
        """
        if self.y_labelencoder is None:
            self.y_labelencoder = LabelEncoder()
            y[:, 0] = self.y_labelencoder.fit_transform(y[:, 0])
        else:
            y[:, 0] = self.y_labelencoder.transform(y[:, 0])
        if categorical:
            y = to_categorical(y)
        self.categorical = categorical
        return y, self.y_labelencoder

    def decode_classifications(self, y, categorical=False):
        """
        Decodes the classifications.

        :param y:  The preprocessed data as a DataFrame.
        :param categorical:  Set to True if categorical was used for encoding.
        Leave false to use the value used for encoding.
        :return:  The decoded data y.
        """
        if self.y_labelencoder is not None:
            if self.categorical or categorical:
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

    @staticmethod
    def cross_fold_validation_scikitlearn(classifier, X, y, cv=10, n_jobs=-1):
        """
        Calculates the cross fold validation mean and variance of Scikit Learn models.

        :param classifier:  The function that builds the classifier.
        :param X:  The X training data.
        :param y:  The y training data.
        :param cv:  The number of cfv groups.
        :param n_jobs:  The number of jobs.  Use -1 to use all CPU cores.
        :return:  A tuple of accuracies, mean, variance, classifiers, confusion matrices.
        """
        cvkfold = StratifiedKFold(n_splits=cv)

        Y = column_or_1d(y)

        classifiers = list()
        accuracies = list()
        cms = list()
        fold = 0
        for train, test in cvkfold.split(X, Y.tolist()):
            fold += 1
            print("Cross Fold Validation - Fold {0}".format(fold))
            classifier = classifier.fit(X[train], Y[train].tolist())
            # probas_ = classifier.predict_proba(X[test])
            # print(probas_)
            y_pred = classifier.predict(X[test])
            accuracy, cm = ML.confusion_matrix_scikitlearn(Y[test], y_pred)
            classifiers.append(classifier)
            accuracies.append(accuracy)
            cms.append(cm)

        accuracies = np.array(accuracies)
        mean = accuracies.mean()
        variance = accuracies.std()
        return accuracies, mean, variance, classifiers, cms

        # Original code here...
        #
        # accuracies = cross_val_score(estimator=classifier,
        #                              X=X,
        #                              y=column_or_1d(y).tolist(),
        #                              cv=cv,
        #                              n_jobs=n_jobs)
        # mean = accuracies.mean()
        # variance = accuracies.std()
        # return accuracies, mean, variance

    @staticmethod
    def cross_fold_validation_keras(classifier_fn, X, y,
                                    batch_size = 10, epochs=100,
                                    cv=10, n_jobs=-1):
        """
        Calculates the cross fold validation mean and variance of Keras models.

        :param classifier_fn:  The function that builds the classifier.
        :param X:  The X training data.
        :param y:  The y training data.
        :param batch_size:  The batch size.
        :param epochs:  The number of epochs.
        :param cv:  The number of cfv groups.
        :param n_jobs:  The number of jobs.  Use -1 to use all CPU cores.
        :return:  A tuple of accuracies, mean, and variance.
        """
        keras_classifier = KerasClassifier(build_fn=classifier_fn,
                                     batch_size=batch_size,
                                     epochs=epochs)
        accuracies = cross_val_score(estimator=keras_classifier,
                                     X=X,
                                     y=y,
                                     cv=cv,
                                     n_jobs=n_jobs)
        mean = accuracies.mean()
        variance = accuracies.std()
        return accuracies, mean, variance

    def plot_roc_curves(self, y_test, y_pred, n_classes=[0, 1, 2, 3, 4, 5]):
        # Compute micro-average ROC curve and ROC area
        yt = label_binarize(y_test.tolist(), classes=n_classes)
        yp = label_binarize(y_pred.tolist(), classes=n_classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in n_classes:
            fpr[i], tpr[i], _ = roc_curve(yt[:, i], yp[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(yt.ravel(), yp.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in n_classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in n_classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(n_classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        lw = 2
        for i, color in zip(n_classes,
                            ['aqua', 'darkorange', 'cornflowerblue', 'red',
                             'green', 'yellow']):
            class_name = self.decode_classifications([i])[0]
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
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
