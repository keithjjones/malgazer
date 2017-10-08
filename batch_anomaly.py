import argparse
from library.files import FileObject
from library.plots import ScatterPlot
import time
import os
import sqlite3
from collections import defaultdict


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates anomalies of entropy files.')
    parser.add_argument('SQLFile',
                        help='The SQLite file for entropy calculations.'
                             '')

    args = parser.parse_args()

    main_conn = sqlite3.connect(args.SQLFile)
    main_cursor = main_conn.cursor()
    sql = "SELECT * FROM metadata;"
    params = {}
    main_cursor.execute(sql, params)
    metadata_results = main_cursor.fetchone()

    while metadata_results:
        dbfile_entropy = metadata_results[7]
        dbfile_anomaly = os.path.splitext(dbfile_entropy)[0]+'_anomaly.db'
        metadata_results = main_cursor.fetchone()

        entropy_conn = sqlite3.connect(dbfile_entropy)
        entropy_cursor = entropy_conn.cursor()
        sql = "SELECT * FROM windowentropy order by offset;"
        params = {}
        entropy_cursor.execute(sql, params)
        entropy_results = entropy_cursor.fetchone()

        entropy_values = dict()

        while entropy_results:
            windowsize = entropy_results[1]
            offset = entropy_results[2]
            entropy = entropy_results[3]

            if windowsize not in entropy_values:
                entropy_values[windowsize] = list()

            entropy_values[windowsize].append(entropy)
            entropy_results = entropy_cursor.fetchone()

        print(entropy_values)

        entropy_conn.close()

    main_conn.close()


if __name__ == "__main__":
    main()
