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
        description='Calculates the entropy of a file.')
    parser.add_argument('MalwareDirectory',
                        help='The directory containing malware to analyze.')
    parser.add_argument('SQLFile',
                        help='The SQLite file to be created for '
                             'the file metadata.')
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

    print("Storing data in SQLite file {0}".format(args.SQLFile))
    # try:
    #     os.remove(args.SQLFile)
    # except:
    #     pass

    main_conn = sqlite3.connect(args.SQLFile)
    main_cursor = main_conn.cursor()

    main_cursor.execute('CREATE TABLE IF NOT EXISTS metadata(' +
                        'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                        'filepath TEXT NOT NULL,'
                        'filesize INT NOT NULL,'
                        'filetype TEXT,'
                        'fileentropy REAL,'
                        'MD5 TEXT,'
                        'SHA256 TEXT,'
                        'DBFile TEXT'
                        ');')
    main_conn.commit()

    # Crawl the directories for malware
    for root, dirs, files in os.walk(args.MalwareDirectory):
        samples_processed = 0
        for f in files:
            # Create the malware file name...
            malwarepath = os.path.join(root, f)
            try:
                m = FileObject(malwarepath)
            except:
                continue

            if m.parsedfile is not None and m.parsedfile['type'] == 'pefile':
                print("\tCalculating: {0} Type: {1}".format(m.filename, m.filetype))

                # Create the DB file name by first creating the directory...
                dbfile_root = root.rstrip(os.path.sep)+"_db"
                dbfile = os.path.join(dbfile_root, f)
                dbfile = dbfile + ".db"

                # Create the directory if needed...
                try:
                    os.stat(dbfile_root)
                except:
                    os.mkdir(dbfile_root)

                sql = "SELECT COUNT(*) FROM metadata where md5 = :md5;"
                params = {'md5': m.md5}
                main_cursor.execute(sql, params)
                results = main_cursor.fetchone()

                # Calculate the entropy of the file...
                fileentropy = m.entropy(normalize)

                if results[0] <= 0:
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
                    # Prepare and execute SQL for sample DB...
                    # try:
                    #     os.remove(dbfile)
                    # except:
                    #     pass
                    malware_conn = sqlite3.connect(dbfile)
                    malware_cursor = malware_conn.cursor()
                    malware_cursor.execute('CREATE TABLE IF NOT EXISTS metadata(' +
                                           'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                                           'filepath TEXT NOT NULL,'
                                           'filesize INT NOT NULL,'
                                           'filetype TEXT,'
                                           'fileentropy REAL,'
                                           'MD5 TEXT,'
                                           'SHA256 TEXT,'
                                           'DBFile TEXT'
                                           ');')
                    # See if this data is already in the database...
                    sql = "SELECT COUNT(*) FROM metadata;"
                    malware_cursor.execute(sql, params)
                    results = malware_cursor.fetchone()
                    if results[0] <= 0:
                        # Prepare and execute SQL for main DB...
                        sql = "INSERT INTO metadata (filepath, filesize, filetype, " + \
                              "fileentropy, MD5, SHA256, DBFile) VALUES " + \
                              "(:filepath, :filesize, :filetype, :fileentropy, " + \
                              ":md5, :sha256, :dbfile);"
                        params = {'filepath': m.filename, 'filesize': m.file_size,
                                  'filetype': m.filetype, 'fileentropy': fileentropy,
                                  'md5': m.md5, 'sha256': m.sha256, 'dbfile': dbfile}
                        malware_cursor.execute(sql, params)
                        malware_conn.commit()
                    malware_cursor.execute(
                        'CREATE TABLE IF NOT EXISTS windowentropy(' +
                        'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                        'windowsize INT NOT NULL,'
                        'offset INT NOT NULL,'
                        'entropy REAL NOT NULL'
                        ');')
                    malware_conn.commit()
                    malware_cursor.execute('CREATE TABLE IF NOT EXISTS windows(' +
                                           'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                                           'windowsize INT NOT NULL,'
                                           'normalized INT NOT NULL'
                                           ');')
                    malware_conn.commit()

                    for w in windows:
                        if w < m.file_size:
                            print("\t\tCalculating window size {0}".format(w))

                            # See if this data is already in the database...
                            sql = "SELECT COUNT(*) FROM windows WHERE windowsize=:windowsize;"
                            params = {'windowsize': w}
                            malware_cursor.execute(sql, params)
                            results = malware_cursor.fetchone()

                            if results[0] == 0:
                                # Calculate running entropy...
                                running_entropy = m.running_entropy(w, normalize)

                                # Delete any old runs...
                                sql = "DELETE FROM windowentropy WHERE windowsize=:windowsize;"
                                params = {'windowsize': w}
                                malware_cursor.execute(sql, params)
                                malware_conn.commit()

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
                                sql = "INSERT INTO windows (windowsize, normalized) " + \
                                      "VALUES (:windowsize, :normalized)"
                                params = {'windowsize': w, 'normalized': normalize}
                                malware_cursor.execute(sql, params)
                                malware_conn.commit()
                            else:
                                print("\t\t\tAlready calculated...")

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
