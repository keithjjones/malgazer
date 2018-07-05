from library.utils import Utils
import pandas as pd
import os
import argparse
import sys


pd.set_option('max_colwidth', 64)


def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the RWE features from a directory of files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('DataDirectory',
                        help='The directory containing the data files.'
                        'This will also contain hashes.txt when finished.')
    parser.add_argument("-p", "--perclass",
                        help="The maximum number of samples, per class."
                             "", type=int, default=10000)
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "", type=int, default=256)
    parser.add_argument("-d", "--datapoints",
                        help="The number of data points to sample running window entropy."
                             "Multiple datapoints can be identified as comma "
                             "separated values without spaces."
                             "", type=int, default=1024)
    if isinstance(arguments, list):
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()

    print("Reading data...")
    # Load data
    all_data, raw_data, classifications = Utils.load_features(args.DataDirectory, 'rwe',
                                                              windowsize=args.window,
                                                              datapoints=args.datapoints)

    print("Trimming data...")
    # Pick samples from each classification
    trimmed_data = all_data.groupby('classification').head(args.perclass)

    print("Saving hashes to hashes.txt...")
    pd.DataFrame(trimmed_data.index).to_csv(os.path.join(args.DataDirectory, 'hashes.txt'), header=False, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)