import argparse
from library.files import FileObject
import numpy
import os
import sqlite3
import time
import shutil


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the entropy of a file.')
    parser.add_argument('MalwareDirectory',
                        help='The directory containing malware to analyze.')
    parser.add_argument('DBDirectory',
                        help='The directory that will contain the Sqlite DB files.')
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "Multiple windows can be identified as comma "
                             "separated values without spaces."
                             "", type=str, required=False)
    parser.add_argument("-n", "--nonormalize", action='store_true',
                        help="Disables entropy normalization."
                             "", required=False)
    parser.add_argument("-m", "--maxsamples", type=int, default=0,
                        help="Maximum number of samples to process, zero for all."
                             "", required=False)
    args = parser.parse_args()

    # Normalize setup...
    if args.nonormalize:
        normalize = False
    else:
        normalize = True

    # Find window sizes
    windows = None
    if args.window:
        windows = args.window.split(',')
        windows = [x.strip() for x in windows]
        windows = [int(x) for x in windows]

    # Delete old data...
    try:
        shutil.rmtree(args.DBDirectory, ignore_errors=True)
    except:
        pass

    # Create DB director...
    try:
        os.mkdir(args.DBDirectory)
    except:
        pass

    # # Create the DB directory if needed...
    # try:
    #     os.stat(args.DBDirectory)
    # except:
    #     os.mkdir(args.DBDirectory)

    print("Storing data in SQLite file {0}".format(os.path.join(args.DBDirectory, 'malgazer.db')))

    main_conn = sqlite3.connect(os.path.join(args.DBDirectory, 'malgazer.db'))
    main_cursor = main_conn.cursor()

    main_cursor.execute('CREATE TABLE IF NOT EXISTS metadata(' +
                        'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                        'filepath TEXT NOT NULL,'
                        'filesize INT NOT NULL,'
                        'filetype TEXT NOT NULL,'
                        'fileentropy REAL NOT NULL,'
                        'MD5 TEXT NOT NULL,'
                        'SHA256 TEXT NOT NULL,'
                        'DBFile TEXT NOT NULL'
                        ');')
    main_conn.commit()

    # Crawl the directories for malware
    samples_processed = 0
    for root, dirs, files in os.walk(args.MalwareDirectory):
        for f in files:
            subdir = root[len(args.MalwareDirectory):]

            # Create the malware file name...
            malwarepath = os.path.join(root, f)
            try:
                m = FileObject(malwarepath)
            except:
                continue

            print("\tCalculating: {0} Type: {1}".format(m.filename, m.filetype))

            # Create the DB file name...
            dbdir = os.path.join(args.DBDirectory, subdir)
            dbfile = os.path.join(dbdir, f) + ".db"

            print("\tSaving data to {0}".format(dbfile))

            # Create the directory if needed...
            try:
                os.stat(dbdir)
            except:
                os.mkdir(dbdir)

            # Calculate the entropy of the file...
            fileentropy = m.entropy(normalize)

            # Prepare and execute SQL for main DB...
            sql = "INSERT INTO metadata (filepath, filesize, filetype, " + \
                  "fileentropy, MD5, SHA256, DBFile) VALUES " + \
                  "(:filepath, :filesize, :filetype, :fileentropy, " + \
                  ":md5, :sha256, :dbfile);"
            params = {'filepath': m.filename, 'filesize': m.file_size,
                      'filetype': m.filetype, 'fileentropy': fileentropy,
                      'md5': m.md5, 'sha256': m.sha256, 'dbfile': dbfile}
            main_cursor.execute(sql, params)
            main_conn.commit()

            # Calculate the window entropy for malware samples...
            if windows is not None:
                # Create malware table structure...
                malware_conn = sqlite3.connect(dbfile)
                malware_cursor = malware_conn.cursor()
                malware_cursor.execute('CREATE TABLE IF NOT EXISTS metadata(' +
                                       'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                                       'filepath TEXT NOT NULL,'
                                       'filesize INT NOT NULL,'
                                       'filetype TEXT NOT NULL,'
                                       'fileentropy REAL NOT NULL,'
                                       'MD5 TEXT NOT NULL,'
                                       'SHA256 TEXT NOT NULL,'
                                       'DBFile TEXT NOT NULL'
                                       ');')

                # Prepare and execute SQL for malware DB...
                sql = "INSERT INTO metadata (filepath, filesize, filetype, " + \
                      "fileentropy, MD5, SHA256, DBFile) VALUES " + \
                      "(:filepath, :filesize, :filetype, :fileentropy, " + \
                      ":md5, :sha256, :dbfile);"
                params = {'filepath': m.filename, 'filesize': m.file_size,
                          'filetype': m.filetype, 'fileentropy': fileentropy,
                          'md5': m.md5, 'sha256': m.sha256, 'dbfile': dbfile}
                malware_cursor.execute(sql, params)
                malware_conn.commit()

                # Create table for window entropy in malware db...
                malware_cursor.execute(
                    'CREATE TABLE IF NOT EXISTS windowentropy(' +
                    'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                    'windowsize INT NOT NULL,'
                    'offset INT NOT NULL,'
                    'entropy REAL NOT NULL'
                    ');')
                malware_conn.commit()

                # Create table for window sizes in malware db...
                malware_cursor.execute('CREATE TABLE IF NOT EXISTS windows(' +
                                       'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                                       'windowsize INT NOT NULL,'
                                       'normalized INT NOT NULL,'
                                       'calctime REAL NOT NULL'
                                       ');')
                malware_conn.commit()

                # Iterate through the window sizes...
                for w in windows:
                    if w < m.file_size:
                        print("\t\tCalculating window size {0:,}".format(w))

                        # Capture the running time
                        start_time = time.time()

                        # Calculate running entropy...
                        running_entropy = m.running_entropy(w, normalize)

                        # Capture the running time
                        end_time = time.time()

                        # Add running entropy to the database...
                        malware_offset = 0
                        for r in running_entropy:
                            sql = "INSERT INTO windowentropy " + \
                                  "(windowsize, offset, entropy) " + \
                                  "VALUES (:windowsize, :offset, :entropy);"
                            params = {'windowsize': w,
                                      'offset': malware_offset,
                                      'entropy': r}
                            malware_cursor.execute(sql, params)
                            malware_offset += 1

                        # Add the window size to the database signifying it is done...
                        sql = "INSERT INTO windows (windowsize, normalized, calctime) " + \
                              "VALUES (:windowsize, :normalized, :calctime)"
                        params = {'windowsize': w, 'normalized': normalize,
                                  'calctime': end_time-start_time}
                        malware_cursor.execute(sql, params)

                # Commit all our data...
                malware_conn.commit()
                malware_conn.close()

            samples_processed += 1
            print("{0:n} samples processed...".format(samples_processed))
            if args.maxsamples > 0 and samples_processed >= args.maxsamples:
                break

    main_conn.commit()
    main_conn.close()


if __name__ == "__main__":
    main()
