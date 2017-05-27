import argparse
from library.files import FileObject
from library.plots import ScatterPlot
import numpy
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

    args = parser.parse_args()

    normalize = True

    print("Storing data in SQLite file {0}".format(args.SQLFile))
    try:
        os.remove(args.SQLFile)
    except:
        pass

    conn = sqlite3.connect(args.SQLFile)
    cursor = conn.cursor()

    cursor.execute('CREATE TABLE metadata(' +
                   'ID INTEGER PRIMARY KEY AUTOINCREMENT,' 
                   'filepath TEXT NOT NULL,'
                   'filesize INT NOT NULL,'
                   'filetype TEXT,'
                   'fileentropy REAL,'
                   'MD5 TEXT,'
                   'SHA256 TEXT,'
                   'DBFile TEXT'
                   ');')
    conn.commit()

    # Crawl the directories for malware
    for root, dirs, files in os.walk(args.MalwareDirectory):
        for f in files:
            malwarepath = os.path.join(root, f)
            m = FileObject(malwarepath)
            print("\tCalculating {0}".format(m.filename))
            fileentropy = m.entropy(normalize)
            dbfile = m.filename + ".db"
            sql = "INSERT INTO metadata (filepath, filesize, filetype, fileentropy, MD5, SHA256, DBFile) VALUES (:filepath, :filesize, :filetype, :fileentropy, :md5, :sha256, :dbfile);"
            params = {'filepath': m.filename, 'filesize':m.file_size,
                    'filetype':m.filetype, 'fileentropy': fileentropy,
                    'md5': m.md5, 'sha256': m.sha256, 'dbfile': dbfile}
            cursor.execute(sql, params)
            conn.commit()

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()