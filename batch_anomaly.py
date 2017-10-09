import argparse
from library.anomaly import AnomalyDetector
import os
import sqlite3


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates anomalies of entropy files.')
    parser.add_argument('SQLFile',
                        help='The SQLite file for entropy calculations.'
                             '')
    parser.add_argument("-r", "--referencesize", type=int, default=2,
                        help="Reference size for anomaly detection."
                             "", required=True)
    parser.add_argument("-p", "--patternsize", type=int, default=5,
                        help="Pattern size for anomaly detection."
                             "", required=True)

    args = parser.parse_args()

    main_conn = sqlite3.connect(args.SQLFile)
    main_cursor = main_conn.cursor()
    sql = "SELECT * FROM metadata;"
    params = {}
    main_cursor.execute(sql, params)
    metadata_results = main_cursor.fetchone()

    while metadata_results:
        dbfile_entropy = metadata_results[7]
        dbfile_anomaly_dir = os.path.split(dbfile_entropy)[0]+"_anomaly"
        dbfile_anomaly = os.path.join(dbfile_anomaly_dir,
                                      os.path.split(os.path.splitext(dbfile_entropy)[0])[1])
        dbfile_anomaly += "_anomaly.db"
        metadata_results = main_cursor.fetchone()

        entropy_conn = sqlite3.connect(dbfile_entropy)
        entropy_cursor = entropy_conn.cursor()
        sql = "SELECT * FROM windowentropy order by offset;"
        params = {}
        entropy_cursor.execute(sql, params)
        entropy_results = entropy_cursor.fetchone()

        entropy_values = dict()
        entropy_offsets = dict()
        entropy_anomalies = dict()

        while entropy_results:
            windowsize = entropy_results[1]
            offset = entropy_results[2]
            entropy = entropy_results[3]

            if windowsize not in entropy_values:
                entropy_values[windowsize] = list()
            if windowsize not in entropy_offsets:
                entropy_offsets[windowsize] = list()

            entropy_values[windowsize].append(entropy)
            entropy_offsets[windowsize].append(offset)

            entropy_results = entropy_cursor.fetchone()

        entropy_conn.close()

        for window in entropy_values:
            entropy_anomalies[window] = AnomalyDetector(entropy_values[window],
                                                        args.referencesize,
                                                        args.patternsize).calculate()

        # Create the directory if needed...
        try:
            os.stat(dbfile_anomaly_dir)
        except:
            os.mkdir(dbfile_anomaly_dir)

        # Create the anomaly DB
        print("Writing {0}".format(dbfile_anomaly))
        anomaly_conn = sqlite3.connect(dbfile_anomaly)
        anomaly_cursor = main_conn.cursor()

        anomaly_cursor.execute('CREATE TABLE IF NOT EXISTS anomaly(' +
                               'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                               'offset INT NOT NULL,'
                               'windowsize INT NOT NULL,'
                               'anomaly REAL'
                               ');')
        anomaly_conn.commit()

        for window in entropy_anomalies:
            for anomaly in entropy_anomalies[window]:
                off = anomaly[0]
                diff = anomaly[1]
                # Prepare and execute SQL for anomaly DB...
                sql = "INSERT INTO anomaly (offset, windowsize, anomaly) " + \
                      "VALUES " + \
                      "(:offset, :windowsize, :anomaly); "
                params = {'offset': off, 'windowsize': window,
                          'anomaly': diff}
                anomaly_cursor.execute(sql, params)
                anomaly_conn.commit()

        anomaly_conn.commit()
        anomaly_conn.close()

    main_conn.close()


if __name__ == "__main__":
    main()
