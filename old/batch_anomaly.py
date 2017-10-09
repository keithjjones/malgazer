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
        filepath = metadata_results[1]
        filesize = metadata_results[2]
        filetype = metadata_results[3]
        fileentropy = metadata_results[4]
        file_md5 = metadata_results[5]
        file_sh256 = metadata_results[6]
        file_dbfile = metadata_results[7]
        dbfile_entropy = file_dbfile
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
        anomaly_cursor = anomaly_conn.cursor()

        anomaly_cursor.execute('CREATE TABLE IF NOT EXISTS anomaly(' +
                               'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                               'offset INT NOT NULL,'
                               'windowsize INT NOT NULL,'
                               'referencesize INT NOT NULL,'
                               'patternsize INT NOT NULL,'
                               'anomaly REAL NOT NULL'
                               ');')
        anomaly_cursor.execute('CREATE TABLE IF NOT EXISTS metadata(' +
                               'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                               'filepath TEXT NOT NULL,'
                               'filesize INT NOT NULL,'
                               'filetype TEXT NOT NULL,'
                               'fileentropy REAL NOT NULL,'
                               'MD5 TEXT NOT NULL,'
                               'SHA256 TEXT NOT NULL,'
                               'DBFile TEXT NOT NULL'
                               ');')
        anomaly_cursor.execute('CREATE TABLE IF NOT EXISTS anomalyinfo(' +
                               'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                               'referencesize INT NOT NULL,'
                               'patternsize INT NOT NULL'
                               ');')
        anomaly_conn.commit()

        # See if this data is already in the database...
        sql = "SELECT COUNT(*) FROM metadata;"
        params = {}
        anomaly_cursor.execute(sql, params)
        results = anomaly_cursor.fetchone()

        if results[0] > 0:
            sql = "INSERT INTO metadata (filepath, filesize, filetype, " + \
                  "fileentropy, MD5, SHA256, DBFile) VALUES " + \
                  "(:filepath, :filesize, :filetype, :fileentropy, " + \
                  ":md5, :sha256, :dbfile);"
            params = {'filepath': filepath, 'filesize': filesize,
                      'filetype': filetype, 'fileentropy': fileentropy,
                      'md5': file_md5, 'sha256': file_sh256, 'dbfile': file_dbfile}
            anomaly_cursor.execute(sql, params)
            anomaly_conn.commit()

        # See if this data is already in the database...
        sql = "SELECT COUNT(*) FROM anomalyinfo WHERE referencesize=:referencesize AND patternsize=:patternsize;"
        params = {'referencesize': args.referencesize,
                  'patternsize': args.patternsize}
        anomaly_cursor.execute(sql, params)
        results = anomaly_cursor.fetchone()

        if results[0] > 0:
            sql = "INSERT INTO anomalyinfo (referencesize, patternsize) " + \
                  "VALUES " + \
                  "(:referencesize, :patternsize);"
            params = {'referencesize': args.referencesize,
                      'patternsize': args.patternsize}
            anomaly_cursor.execute(sql, params)
            anomaly_conn.commit()

        for window in entropy_anomalies:
            for anomaly in entropy_anomalies[window]:
                off = anomaly[0]
                diff = anomaly[1]
                # Prepare and execute SQL for anomaly DB...
                sql = "INSERT INTO anomaly (offset, windowsize, " \
                      "referencesize, patternsize, anomaly) " + \
                      "VALUES " + \
                      "(:offset, :windowsize, :referencesize, :patternsize, :anomaly); "
                params = {'offset': off, 'windowsize': window,
                          'anomaly': diff, 'referencesize': args.referencesize,
                          'patternsize': args.patternsize}
                anomaly_cursor.execute(sql, params)
                anomaly_conn.commit()

        anomaly_conn.commit()
        anomaly_conn.close()

    main_conn.close()


if __name__ == "__main__":
    main()
