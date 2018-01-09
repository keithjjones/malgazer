# Script to calculate the entropy of a file already calculated with the batch script.
import argparse
import sqlite3
from library.files import FileObject
from library.plots import ScatterPlot
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import numpy
import time


def main():
    # Capture the running time
    start_time = time.time()

    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the entropy of a file calculated from batch processing.')
    parser.add_argument('SQLFile',
                        help='The malware SQL file to analyze.')
    parser.add_argument("-pre", "--plotrunningentropy", action='store_true',
                        help="Plot the running entropy values.  Only valid "
                             "if -w is used!"
                             "", required=False)
    parser.add_argument("-preskip", "--plotrunningentropyskip",
                        help="Skip this number of bytes in running entropy "
                             "value plot for compression."
                             "", default=1, type=int, required=False)
    parser.add_argument("-prefilename", "--plotrunningentropyfilename",
                        help="The file name of the output file for a running "
                             "entropy plot (html).",
                        default="malgzer_running_entropy.html",
                        required=False)
    parser.add_argument("-a", "--anomaly", action='store_true',
                        help="Enable anomaly detection."
                             "", required=False)
    parser.add_argument("-c", "--contamination", type=float, default=0.1,
                        help="Outlier contamination factor."
                             "", required=False)
    parser.add_argument("-l", "--lofneighbors", type=int, default=300,
                        help="Local outlier factor neighbors."
                             "", required=False)

    args = parser.parse_args()


    main_conn = sqlite3.connect(args.SQLFile)
    main_cursor = main_conn.cursor()

    # Find the windows calculated...
    main_cursor.execute('SELECT * from windows;')
    results = main_cursor.fetchone()
    windows = []

    while (results):
        windows.append(results[1])
        results = main_cursor.fetchone()

    # Setup running entropy variables...
    running_entropy = dict()
    for w in windows:
        running_entropy[w] = list()

    # Read in running entropy values...
    main_cursor.execute('SELECT * from windowentropy ORDER BY offset;')
    results = main_cursor.fetchone()

    while(results):
        running_entropy[results[1]].append(results[3])
        results = main_cursor.fetchone()

    # Print the running time
    print()
    print("Total running time: {0:.6f} seconds"
          .format(round(time.time() - start_time, 6)))
    print()


if __name__ == "__main__":
    main()
