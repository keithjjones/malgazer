import argparse
import os
import sqlite3
from library.anomaly import AnomalyDetector


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Detects anomalies in the entropy data.')
    parser.add_argument('MalwareEntropyDB',
                        help='The SQLite DB for the entropy of a malware file.')

    args = parser.parse_args()

    conn = sqlite3.connect(args.MalwareEntropyDB)
    cursor = conn.cursor()

    sql = "SELECT * FROM windowentropy ORDER BY offset ASC;"

    cursor.execute(sql, {})

    # Store our results
    results = dict()

    for row in cursor:
        if row[1] not in results:
            results[row[1]] = dict()
            results[row[1]]['offset'] = list()
            results[row[1]]['entropy'] = list()
        results[row[1]]['offset'].append(row[2])
        results[row[1]]['entropy'].append(row[3])

    print(results.keys())


if __name__ == "__main__":
    main()