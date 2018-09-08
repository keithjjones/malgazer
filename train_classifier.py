import sys
import argparse
import time
import os
import pandas as pd
import numpy as np
import json
from library.utils import Utils
from library.ml import ML, Keras_Dense_Parameters, Keras_Conv1D_Parameters, \
    Keras_Flatten_Parameters, Keras_Dropout_Parameters


pd.set_option('max_colwidth', 64)

# This controls the size and look of the divider
DIVIDER_LENGTH = 80
DIVIDER_CHAR = '='
DIVIDER = DIVIDER_CHAR*DIVIDER_LENGTH


CLASSIFIER_TO_STRING = {
    'adaboost': 'AdaBoost (adaboost)',
    'ann': 'Artificial Neural Network (ann)',
    'cnn': 'Convolutional Neural Network (cnn)',
    'dt': 'Decision Tree (dt)',
    'gridsearch': 'Grid Search (gridsearch)',
    'knn': 'k-Nearest Neighbors (knn)',
    'nb': 'Naive Bayes (nb)',
    'nc': 'Nearest Centroid (nc)',
    'ovr': 'OneVRest (ovr)',
    'rf': 'Random Forest (rf)',
    'svm': 'Support Vector Machine (svm)'
}


def get_estimator_static(classifier_type, *args, **kwargs):
    """
    Returns a static estimator based on classifier type.

    :param classifier_type:  A valid classifier type.
    :param args:  Args to be passed to base constructor.
    :param kwargs:  Kwargs to be passed to the base constructor.
    :return:  The classifier
    """
    if classifier_type.lower() == 'svm':
        return ML.build_svm_static(*args, **kwargs)
    elif classifier_type.lower() == 'dt':
        return ML.build_dt_static(*args, **kwargs)
    elif classifier_type.lower() == 'nb':
        return ML.build_nb_static(*args, **kwargs)
    elif classifier_type.lower() == 'rf':
        return ML.build_rf_static(*args, **kwargs)
    elif classifier_type.lower() == 'knn':
        return ML.build_knn_static(*args, **kwargs)
    elif classifier_type.lower() == 'nc':
        return ML.build_nc_static(*args, **kwargs)
    elif classifier_type.lower() == 'adaboost':
        return ML.build_adaboost_static(*args, **kwargs)
    elif classifier_type.lower() == 'ovr':
        return ML.build_ovr_static(*args, **kwargs)
    elif classifier_type.lower() == 'ann':
        from keras.wrappers.scikit_learn import KerasClassifier
        def create_model():
            return ML.build_ann_static(*args, **kwargs)
        return KerasClassifier(build_fn=create_model)
    elif classifier_type.lower() == 'cnn':
        from keras.wrappers.scikit_learn import KerasClassifier
        def create_model():
            return ML.build_cnn_static(*args, **kwargs)
        return KerasClassifier(build_fn=create_model)
    else:
        return None


def get_estimator(classifier_type, ml, *args, **kwargs):
    """
    Returns an estimator based on classifier type.

    :param classifier_type:  A valid classifier type.
    :param ml:  The ml object to create an estimator.
    :param args:  Args to be passed to base constructor.
    :param kwargs:  Kwargs to be passed to the base constructor.
    :return:  The classifier
    """
    if classifier_type.lower() == 'svm':
        return ml.build_svm(*args, **kwargs)
    elif classifier_type.lower() == 'dt':
        return ml.build_dt(*args, **kwargs)
    elif classifier_type.lower() == 'nb':
        return ml.build_nb(*args, **kwargs)
    elif classifier_type.lower() == 'rf':
        return ml.build_rf(*args, **kwargs)
    elif classifier_type.lower() == 'knn':
        return ml.build_knn(*args, **kwargs)
    elif classifier_type.lower() == 'nc':
        return ml.build_nc(*args, **kwargs)
    elif classifier_type.lower() == 'adaboost':
        return ml.build_adaboost(*args, **kwargs)
    elif classifier_type.lower() == 'ovr':
        return ml.build_ovr(*args, **kwargs)
    elif classifier_type.lower() == 'ann':
        return ml.build_ann(*args, **kwargs)
    elif classifier_type.lower() == 'cnn':
        return ml.build_cnn(*args, **kwargs)
    else:
        return None


def create_base_estimator_params(extra_estimator_params, search_phrase="estimator__"):
    """
    A function to create the extra and base estimator parameters for the classifiers that
    require a base estimator.

    :param extra_estimator_params:  The total params
    :param search_phrase:  The search phrase with the base estimator params
    :return: extra_params, base_estimator_params
    """
    keystodel = []
    base_estimator_params = {}
    extra_params = extra_estimator_params
    for key in extra_estimator_params:
        if search_phrase in key:
            newkey = key.split('__')[1]
            base_estimator_params[newkey] = extra_estimator_params[key]
            keystodel.append(key)
    for delkey in keystodel:
        del extra_params[delkey]
    return extra_params, base_estimator_params


def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Trains a malware classifier.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ClassifierType',
                        help='The classifier type.  Valid values: '
                        'ann - Artifical Neural Network, cnn - Convolutional Neural Network, '
                        'dt - Decision Tree, svm - Support Vector Machine, nb - Naive Bayes, '
                        'rf - Random Forest, knn - k-Nearest Neighbors, nc - Nearest Centroid, '
                        'adaboost - Adaboost (requires a base estimator), '
                        'ovr - OneVRest (requires a base estimator), '
                        'gridsearch - Grid search (requires a base estimator and does not save classifier)')
    parser.add_argument('FeatureType',
                        help='The feature type. Valid values: '
                        'rwe - Running Window Entropy, gist - GIST image features')
    parser.add_argument('DataDirectory',
                        help='The directory containing the feature files.')
    parser.add_argument("-t", "--test",
                        help="The percentage of samples used for testing (set to zero to use the whole data set).  "
                             "This should be 0.1 for 10\%%, etc."
                             "", type=float, default=0)
    parser.add_argument("-c", "--crossval",
                        help="The number of groups for cross validation.  Set to zero to disable cross validation.  "
                             "Cross validation will use all the samples if enabled.  "
                             "Not available for gridsearch.", type=int, default=0)
    parser.add_argument("-cj", "--crossvaljobs",
                        help="The number jobs for cross validation."
                             "", type=int, default=10)
    parser.add_argument("-n", "--numclasses",
                        help="The number of classes in the training data."
                             "", type=int, default=6)
    parser.add_argument("-e", "--estimatorparams",
                        help="Extra params to use for an estimator, as a JSON dict.  Leave as '{}' to add nothing.  "
                             "Ignored for gridsearch."
                             "", type=str, default='{}')
    parser.add_argument("-roc", "--roccurves", action='store_true',
                        help="Plot the ROC curves."
                             "", required=False)
    parser.add_argument("-rwew", "--rwewindowsize",
                        help="The window size of RWE to use as features."
                             "", type=int, default=256)
    parser.add_argument("-rwed", "--rwedatapoints",
                        help="The number of datapoints of RWE to use as features."
                             "", type=int, default=1024)
    parser.add_argument("-nnb", "--nnbatchsize",
                        help="The batch size used for training neural networks."
                             "", type=int, default=100)
    parser.add_argument("-nne", "--nnepochs",
                        help="The epochs used for training neural networks."
                             "", type=int, default=10)
    parser.add_argument("-nnl", "--nnlayers",
                        help="The file containing the Python code to instantiate neural network layers.  "
                             "The content should be a list containing a list of layers if you are grid searching."
                             "", type=str, default="")
    parser.add_argument("-gt", "--gridsearchtype",
                        help="The type of the base estimator for gridsearch."
                             "", type=str, default='dt')
    parser.add_argument("-gp", "--gridsearchparams",
                        help="The params for the gridsearch.  This is a JSON string that will be accepted by "
                            "the GridsearchCV in scikit learn."
                             "", type=str, default='{}')
    parser.add_argument("-gj", "--gridsearchjobs",
                        help="The number of jobs for the gridsearch.  -1 uses all available CPU cores."
                             "", type=int, default=-1)
    parser.add_argument("-gc", "--gridsearchcv",
                        help="The number of cross validation groups for the gridsearch."
                             "", type=int, default=5)
    parser.add_argument("-at", "--adaboosttype",
                        help="The type of the base estimator for adaboost."
                             "", type=str, default='dt')
    parser.add_argument("-ot", "--ovrtype",
                        help="The type of the base estimator for ovr."
                             "", type=str, default='dt')
    if isinstance(arguments, list):
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()

    datadir = args.DataDirectory
    feature_type = args.FeatureType.lower()
    classifier_type = args.ClassifierType.lower()
    cross_fold_validation = args.crossval > 0
    cfv_groups = args.crossval
    cfv_jobs = args.crossvaljobs
    generate_roc_curves = args.roccurves
    test_percent = args.test
    windowsize = args.rwewindowsize
    datapoints = args.rwedatapoints
    n_categories = args.numclasses
    batch_size = args.nnbatchsize
    epochs = args.nnepochs
    gridsearch_type = args.gridsearchtype.lower()
    gridsearch_params = json.loads(args.gridsearchparams)
    gridsearch_njobs = args.gridsearchjobs
    gridsearch_cv = args.gridsearchcv
    adaboost_type = args.adaboosttype
    ovr_type = args.ovrtype.lower()
    ovr_base_estimator = get_estimator_static(ovr_type)
    extra_estimator_params = json.loads(args.estimatorparams)
    if len(args.nnlayers.strip()) > 0:
        nnlayers = eval(open(args.nnlayers.strip(), 'r').read())
    else:
        nnlayers = None

    if test_percent < 0 or test_percent > 1:
        raise ValueError("Test percent should be between 0 and 1!")

    # if cross_fold_validation or classifier_type == 'gridsearch':
    #         test_percent = 0

    print(DIVIDER)
    print("Loading data...")
    print(DIVIDER)

    # Load data
    all_data, raw_data, classifications = Utils.load_features(datadir, feature_type, filterhashes=True,
                                                              windowsize=windowsize, datapoints=datapoints)

    print(DIVIDER)
    print("Feature Type: {0}".format(feature_type))
    if feature_type == 'rwe':
        print("\tWindow Size: {0:,}".format(windowsize))
        print("\tData points: {0:,}".format(datapoints))
    print("Number of features: {0:,}".format(len(raw_data.columns)))
    print(DIVIDER)

    # Assemble the final training data
    X = all_data.drop('classification', axis=1).values.copy()
    y = pd.DataFrame(all_data['classification']).values.copy()

    # Make the classifier
    ml = ML(feature_type=feature_type, classifier_type=classifier_type, n_classes=n_categories,
            rwe_windowsize=windowsize, datapoints=datapoints, nnlayers=nnlayers)
    X, y = ml.preprocess_data(X, y)

    # Check for bad values...
    if pd.DataFrame(X).isin([np.nan, np.inf, -np.inf]).any(1).any():
        raise Exception("X has issues.")
    # if pd.DataFrame(y).isin([np.nan, np.inf, -np.inf]).any(1).any():
    #     raise Exception("y has issues.")

    if test_percent > 0:
        X_train, X_test, y_train, y_test = ml.train_test_split(X, y, test_percent=test_percent)
    else:
        X_train = X
        y_train = y
        X_test = X
        y_test = y

    # List the data...
    ytr = ml.decode_classifications(y_train.tolist())
    yte = ml.decode_classifications(y_test.tolist())
    print("\n")
    print(DIVIDER)
    print("Classifier Type: {0}".format(CLASSIFIER_TO_STRING[classifier_type]))
    if classifier_type == 'gridsearch':
        print(DIVIDER)
        print("Grid Search Enabled!")
        print("Grid Search Type: {0}".format(CLASSIFIER_TO_STRING[gridsearch_type]))
        print("Grid Search Parameters: {0}".format(gridsearch_params))
        if gridsearch_type == 'ovr':
            print(DIVIDER)
            print('OneVRest Enabled!')
            print("OneVRest Type: {0}".format(CLASSIFIER_TO_STRING[ovr_type]))
        if gridsearch_type == 'adaboost':
            print(DIVIDER)
            print("AdaBoost Enabled!")
            print("AdaBoost Type: {0}".format(CLASSIFIER_TO_STRING[adaboost_type]))
    if cross_fold_validation:
        print(DIVIDER)
        print("Cross Fold Validation - {0} Fold(s) - Enabled!".format(cfv_groups))
    if classifier_type == 'ovr':
        print(DIVIDER)
        print('OneVRest Enabled!')
        print("OneVRest Type: {0}".format(CLASSIFIER_TO_STRING[ovr_type]))
    if classifier_type == 'adaboost':
        print(DIVIDER)
        print("AdaBoost Enabled!")
        print("AdaBoost Type: {0}".format(CLASSIFIER_TO_STRING[adaboost_type]))
    print(DIVIDER)
    print("Training Class Count:")
    print(DIVIDER)
    print("\t{0}".format(pd.DataFrame(ytr)[0].value_counts().to_string().replace('\n', '\n\t')))
    print(DIVIDER)
    print("Testing Class Count:")
    print(DIVIDER)
    print("\t{0}".format(pd.DataFrame(yte)[0].value_counts().to_string().replace('\n', '\n\t')))
    print(DIVIDER)
    print("\n")

    print(DIVIDER)
    print("Begin training...")
    print(DIVIDER)
    if classifier_type.lower() == 'cnn':
        if cross_fold_validation is False:
            # Create the CNN
            if nnlayers:
                classifier = ml.build_cnn(X_train, y_train, layers=nnlayers)
            else:
                classifier = ml.build_cnn(X_train, y_train)

            # Train the CNN
            start_time = time.time()
            classifier = ml.train(X_train, y_train, batch_size=batch_size, epochs=epochs, tensorboard=False)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print(DIVIDER)
            print("\n")

            # Predict the results
            y_pred = ml.predict(X_test)

            # Making the Confusion Matrix
            accuracy, cm = ml.confusion_matrix(y_test, y_pred)

            print(DIVIDER)
            print("Confusion Matrix:")
            print(cm)
            print(DIVIDER)
            print("\tAccuracy:")
            print("\t{0}".format(accuracy))
            print(DIVIDER)

            if generate_roc_curves:
                ml.plot_roc_curves(y_test, y_pred, n_categories, filename="cnn_{0}.png".format(feature_type))
        else:
            # Cross Fold Validation
            start_time = time.time()
            mean, variance, classifiers = ml.cross_fold_validation(X_train, y_train,
                                                                   batch_size=batch_size,
                                                                   epochs=epochs,
                                                                   cv=cfv_groups, n_jobs=cfv_jobs)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print(DIVIDER)
            print("CFV Mean: {0}".format(mean))
            print("CFV Var: {0}".format(variance))
            print(DIVIDER)

            if generate_roc_curves:
                for fold in range(cfv_groups):
                    ml.set_classifier_by_fold(fold+1)
                    y_test = ml.classifiers[fold+1]['y_test']
                    y_pred = ml.classifiers[fold+1]['y_pred']
                    ml.plot_roc_curves(y_test, y_pred, n_categories, fold+1, filename="cnn-fold-{0}_{1}.png".format(
                        fold+1, feature_type))

    elif classifier_type.lower() == 'ann':
        if cross_fold_validation is False:
            # Create the ANN
            if nnlayers:
                classifier = ml.build_ann(X_train, y_train, layers=nnlayers)
            else:
                classifier = ml.build_ann(X_train, y_train)

            # Train the NN
            start_time = time.time()
            classifier = ml.train(X_train, y_train, batch_size=batch_size, epochs=epochs, tensorboard=False)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print(DIVIDER)
            print("\n")

            # Predict the results
            y_pred = ml.predict(X_test)

            # Making the Confusion Matrix
            accuracy, cm = ml.confusion_matrix(y_test, y_pred)

            print(DIVIDER)
            print("Confusion Matrix:")
            print(cm)
            print(DIVIDER)
            print("\tAccuracy:")
            print("\t{0}".format(accuracy))
            print(DIVIDER)

            if generate_roc_curves:
                ml.plot_roc_curves(y_test, y_pred, n_categories, filename="ann_{0}.png".format(feature_type))
        else:
            # Cross Fold Validation
            start_time = time.time()
            mean, variance, classifiers = ml.cross_fold_validation(X_train, y_train,
                                                                   batch_size=batch_size,
                                                                   epochs=epochs,
                                                                   cv=cfv_groups, n_jobs=cfv_jobs)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print(DIVIDER)
            print("CFV Mean: {0}".format(mean))
            print("CFV Var: {0}".format(variance))
            print(DIVIDER)

            if generate_roc_curves:
                for fold in range(cfv_groups):
                    ml.set_classifier_by_fold(fold+1)
                    y_test = ml.classifiers[fold+1]['y_test']
                    y_pred = ml.classifiers[fold+1]['y_pred']
                    ml.plot_roc_curves(y_test, y_pred, n_categories, fold+1, filename="ann-fold-{0}_{1}.png".format(
                        fold+1, feature_type))
    elif classifier_type.lower() == 'gridsearch':
        Xt = X_train
        yt = y_train

        if gridsearch_type.lower() == 'ann':
            if nnlayers:
                classifier = get_estimator_static(gridsearch_type.lower(), Xt, yt, layers=nnlayers)
            else:
                classifier = get_estimator_static(gridsearch_type.lower(), Xt, yt)
        elif gridsearch_type.lower() == 'cnn':
            if nnlayers:
                classifier = get_estimator_static(gridsearch_type.lower(), Xt, yt)
            else:
                classifier = get_estimator_static(gridsearch_type.lower(), Xt, yt, layers=nnlayers)
        if gridsearch_type.lower() == 'adaboost':
            base_estimator = get_estimator_static(adaboost_type.lower())
            classifier = get_estimator_static(gridsearch_type.lower(), base_estimator=base_estimator)
        if gridsearch_type.lower() == 'ovr':
            base_estimator = get_estimator_static(ovr_type.lower())
            classifier = get_estimator_static(gridsearch_type.lower(), estimator=base_estimator)
        else:
            classifier = get_estimator_static(gridsearch_type.lower())

        classifier = ml.build_gridsearch(gridsearch_type=gridsearch_type, estimator=classifier,
                                         param_grid=gridsearch_params,
                                         cv=gridsearch_cv, n_jobs=gridsearch_njobs)
        start_time = time.time()

        classifier = ml.train(Xt, yt)
        print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
        print(DIVIDER)
        print("\n")

        print(DIVIDER)
        print("Best Score: {0}".format(classifier.best_score_))
        print(DIVIDER)
        print("CV Results:")
        print("{0}".format(classifier.cv_results_))
        print(DIVIDER)
        print("Params:")
        print("{0}".format(classifier.cv_results_['params']))
        print(DIVIDER)
        print("Mean Test Score: {0}".format(classifier.cv_results_['mean_test_score']))
        best_param = classifier.cv_results_['params'][(classifier.cv_results_['mean_test_score'] == classifier.best_score_).argmax()]
        print(DIVIDER)
        print("Best Params: {0}".format(best_param))
        print(DIVIDER)
    else:
        # This area is for scikit learn models
        classifier_short = ""
        if classifier_type.lower() == 'adaboost':
            extra_estimator_params, base_estimator_params = create_base_estimator_params(extra_estimator_params,
                                                                                         "base_estimator__")
            base_estimator = get_estimator_static(adaboost_type.lower(), **base_estimator_params)
            estimator_params = {'base_estimator': base_estimator, 'adaboost_type': adaboost_type}
            classifier_short = "{0}-{1}".format(classifier_type.lower(), adaboost_type.lower())
        elif classifier_type.lower() == 'ovr':
            extra_estimator_params, base_estimator_params = create_base_estimator_params(extra_estimator_params,
                                                                                         "estimator__")
            base_estimator = get_estimator_static(ovr_type.lower(), **base_estimator_params)
            estimator_params = {'estimator': base_estimator, 'ovr_type': ovr_type}
            classifier_short = "{0}-{1}".format(classifier_type.lower(), ovr_type.lower())
        else:
            estimator_params = {}
            classifier_short = "{0}".format(classifier_type.lower())

        estimator_params.update(extra_estimator_params)
        print(DIVIDER)
        print("Model hyperparameters: {0}".format(estimator_params))
        print(DIVIDER)
        classifier = get_estimator(classifier_type.lower(), ml, **estimator_params)

        start_time = time.time()
        if cross_fold_validation is False:
            classifier = ml.train(X_train, y_train)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print(DIVIDER)
            print("\n")
            y_pred = ml.predict(X_test)
            # probas = ml.classifier.predict_proba(X_test)

            # Making the Confusion Matrix
            accuracy, cm = ml.confusion_matrix(y_test, y_pred)

            print(DIVIDER)
            print("Confusion Matrix:")
            print(cm)
            print(DIVIDER)
            print("\tAccuracy:")
            print("\t{0}".format(accuracy))
            print(DIVIDER)

            if generate_roc_curves:
                ml.plot_roc_curves(y_test, y_pred, n_categories, filename="{0}_{1}.png".format(
                    classifier_short, feature_type))
        else:
            # Cross Fold Validation
            mean, variance, classifiers = ml.cross_fold_validation(X_train, y_train,
                                                                   cv=cfv_groups, n_jobs=cfv_jobs)
            print("Training time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
            print(DIVIDER)
            print("CFV Mean: {0}".format(mean))
            print("CFV Var: {0}".format(variance))
            print(DIVIDER)

            if generate_roc_curves:
                for fold in range(cfv_groups):
                    ml.set_classifier_by_fold(fold+1)
                    y_test = ml.classifiers[fold+1]['y_test']
                    y_pred = ml.classifiers[fold+1]['y_pred']
                    ml.plot_roc_curves(y_test, y_pred, n_categories, fold+1,
                                       filename="{0}-fold-{1}_{2}.png".format(classifier_short, fold+1, feature_type))

    # Save the classifier
    print("\n")
    if feature_type == 'rwe':
        path = os.path.join(datadir, "classifiers",
                            "classifiers_rwe_{0}_window_{1}_datapoints".format(windowsize, datapoints),
                            classifier_type.lower())
    else:
        path = os.path.join(datadir, "classifiers", "GIST", classifier_type.lower())

    try:
        os.stat(path)
    except:
        os.makedirs(path)
    if classifier_type.lower() != 'gridsearch' and cfv_groups < 1:
        print(DIVIDER)
        print("Saving the classifier...")
        ml.save_classifier(path)
        print("Classifier saved to: {0}".format(path))
        print(DIVIDER)
        print("\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    print(DIVIDER)
    print("Command Line:")
    print("\t{0}".format(' '.join(sys.argv)))
    print(DIVIDER)
    main(args)
