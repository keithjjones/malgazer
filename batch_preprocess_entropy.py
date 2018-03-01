import argparse
from library.utils import Utils


def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the entropy of a file.')
    parser.add_argument('DataDirectory',
                        help='The directory containing malware data to analyze, calculated by batch_calculate_entropy.')
    parser.add_argument('OutputDirectory',
                        help='The output directory where the data will be saved.')
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "", type=int, default=256, required=False)
    parser.add_argument("-d", "--datapoints",
                        help="The number of data points to sample running window entropy."
                             "", type=int, default=512, required=False)

    if isinstance(arguments, list):
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()


    df = Utils.batch_preprocess_rwe_data(args.DataDirectory,
                                         args.datapoints,
                                         args.window)

    classifications = Utils.get_classifications_from_path(args.DataDirectory)

    # Utils.extract_tsfresh_relevant_features(extracted_features, classifications_ordered['classification'])

    Utils.save_preprocessed_data(df, classifications,
                                 args.OutputDirectory)


if __name__ == "__main__":
    main()
