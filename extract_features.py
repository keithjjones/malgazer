import argparse
from library.files import FileObject
from library.utils import Utils
import numpy
import os
import time
import shutil
import sys


def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the features from a directory of files.')
    parser.add_argument('MalwareDirectory',
                        help='The directory containing malware to analyze.')
    parser.add_argument('DataDirectory',
                        help='The directory that will contain the data files.')
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "", type=int, required=False)
    parser.add_argument("-n", "--nonormalize", action='store_true',
                        help="Disables entropy normalization."
                             "", required=False)
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

    # Normalize setup...
    if args.nonormalize:
        normalize = False
    else:
        normalize = True

    # Crawl the directories for malware and calculate rwe
    Utils.extract_features_from_directory(args.MalwareDirectory,
                                          args.DataDirectory,
                                          window_size=args.window,
                                          normalize=normalize,
                                          njobs=args.jobs)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
