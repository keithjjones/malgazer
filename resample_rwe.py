import argparse
from library.entropy import resample_rwe
import sys
import os

def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Resamples RWE data.')
    parser.add_argument('OriginalFile',
                        help='The file containing RWE for samples.')
    parser.add_argument('OutputDirectory',
                        help='The output directory where the data will be saved.')
    parser.add_argument("-d", "--datapoints",
                        help="The number of data points to sample running window entropy."
                             "", type=int, default=512, required=False)
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="The number of jobs to do the work, but be 1 or greater."
                             "", required=False)

    if isinstance(arguments, list):
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()

    if args.jobs < 1:
        print("Jobs must be 1 or greater.")
        exit()

    resample_rwe(args.OriginalFile, "{0}_{1}.csv".format(os.path.splitext(args.OriginalFile)[0], args.datapoints))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
