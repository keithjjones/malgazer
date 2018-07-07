import sys
import argparse
import pandas as pd

pd.set_option('max_colwidth', 64)


def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Trains a malware classifier.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ClassifierType',
                        help='The classifier type.  Valid values: '
                        'ann - Artifical Neural Network, cnn - Convolutional Neural Network, '
                        'dt - Decision Tree, svm - State Vector Machine, nb - Naive Bayes, '
                        'rf - Random Forest, knn - k-Nearest Neighbors, nc - Nearest Centroid, '
                        'adaboost - Adaboost (requires a base estimator), '
                        'ovr - OneVRest (requires a base estimator), '
                        'gridsearch - Grid search (requires a base estimator and does not save classifier)')
    parser.add_argument('FeatureType',
                        help='The feature type. Valid values: '
                        'rwe - Running Window Entropy, gist - GIST image features')
    parser.add_argument('DataDirectory',
                        help='The directory containing the feature files.')
    if isinstance(arguments, list):
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
