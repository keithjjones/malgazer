# Machine Learning Module
import pandas as pd
import numpy as np
from numpy import argmax
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, InputLayer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras.backend as K


class ML(object):
    def __init__(self):
        super(ML, self).__init__()
        self.classifer = None
        self.X_sc = None

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
        classifier.add(Dense(units=1250, activation='relu'))
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
        self.classifier = ML.build_cnn_static(input, outputs)
        self.classifier.summary()
        return self.classifier

    def train_nn(self, X_train, y_train, batch_size=50, epochs=100):
        """
        Trains a given neural network with X_train and y_train.

        :param classifier:  The classifier to train.
        :param X_train:  The X training data.
        :param y_train:  The y training data.
        :param batch_size:  The batch size.
        :param epochs:  The number of epochs.
        :return: The classifier after training.
        """
        self.classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
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
    def confusion_matrix(y_test, y_pred):
        """
        Calculates the confusion matrix for predictions.

        :param y_test:  The y testing data.
        :param y_pred:  The y predicted data.
        :return:  The accuracty,confusion_matrix, as a tuple.
        """
        cm = confusion_matrix(y_test.argmax(1), y_pred.argmax(1))
        accuracy = 0.
        for i in range(0, len(cm)):
            accuracy += cm[i, i]
        accuracy = accuracy/cm.sum()
        return accuracy, cm

    def scale_features_preprocessed_data(self, X):
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

    @staticmethod
    def encode_preprocessed_data(y):
        """
        Encodes the classifications.

        :param y:  The preprocessed data as a DataFrame.
        :return:  A tuple of the encoded data y and the encoder (for inverting)
        as y,encoder.
        """
        labelencoder_y = LabelEncoder()
        y[:, 0] = labelencoder_y.fit_transform(y[:, 0])
        y = to_categorical(y)
        return y, labelencoder_y

    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=0):
        """
        Creates a training and testing data sets.

        :param X:  The X values as a DataFrame.
        :param y:  The y values as a DataFrame.
        :param test_size: The percentage, as a decimal, of the test data set size.
        :param random_state:  The random seed.
        :return: A tuple of X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def cross_fold_validation(classifier_fn, X_train, y_train,
                              batch_size = 10, epochs=100,
                              cv=10, n_jobs=-1):
        """
        Calculates the cross fold validation mean and variance.

        :param classifier_fn:  The function that builds the classifier.
        :param X_train:  The X training data.
        :param y_train:  The y training data.
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
                                     X=X_train,
                                     y=y_train,
                                     cv=cv,
                                     n_jobs=n_jobs)
        mean = accuracies.mean()
        variance = accuracies.std()
        return accuracies, mean, variance
