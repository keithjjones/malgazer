# Machine Learning Module
import pandas as pd
import numpy as np
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, InputLayer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


class ML(object):
    def __init__(self):
        super(ML, self).__init__()
        self.classifer = None

    def build_ann(self, datapoints=1024):
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
        classifier.add(Dense(units=3,
                             kernel_initializer='uniform',
                             activation='softmax'))
        classifier.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.classifer = classifier
        return classifier

    def build_cnn(self):
        """
        Create a generic CNN.

        :return:  The classifier.
        """
        classifier = Sequential()
        classifier.add(InputLayer(input_shape=Xt.shape[1:]))
        classifier.add(Conv1D(filters=32, kernel_size=50, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=4))
        classifier.add(Conv1D(filters=32, kernel_size=20, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=2))
        classifier.add(Flatten())
        classifier.add(Dense(units=512, activation='relu'))
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=64, activation='relu'))
        classifier.add(Dense(units=3, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.classifer = classifier
        return classifier

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
        for i in len(cm):
            accuracy += cm[i, i]
        accuracy = accuracy/cm.sum()
        return accuracy, cm