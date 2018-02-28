import argparse
from library.utils import Utils


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the entropy of a file.')
    parser.add_argument('DataDirectory',
                        help='The directory containing malware data to analyze, calculated by batch_calculate_entropy.')
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "", type=str, required=False)
    parser.add_argument("-d", "--datapoints",
                        help="The number of data points to sample running window entropy."
                             "", type=str, required=False)
    args = parser.parse_args()

    extracted_features,df = Utils.batch_tsfresh_rwe_data(args.DataDirectory,
                                                         args.datapoints,
                                                         args.window)
    classifications = Utils.get_classifications_from_path(args.DataDirectory)
    classifications_ordered = Utils.create_ordered_classifications(classifications,
                                                                   extracted_features)
    Utils.save_processed_data(df, classifications,
                              classifications_ordered,
                              extracted_features, "data")


if __name__ == "__main__":
    main()
