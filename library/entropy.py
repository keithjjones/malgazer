# Entropy Module
import math
import sqlite3
import os
import json
import time
from collections import Counter
from collections import deque
from scipy import interpolate
import numpy as np


class RunningEntropy(object):
    """Class to hold running entropy calculations."""
    def __init__(self):
        """
        Class to hold running entropy calculations.

        :return:  Nothing.
        """
        # Each of these are keyed by window size
        self.entropy_data = dict()
        self.normalize = dict()
        self.calctime = dict()

        # This can hold metadata info
        self.metadata = dict()

    def calculate(self, inputbytes, window=256, normalize=True):
        """
        Calculates the running entropy of inputbytes with the window size.
        This algorithm is optimized for single thread speed.

        :param inputbytes: A list of values representing bytes.
        :param window:  The size, in bytes, of the window to calculate for
            running entropy.  Must be larger than one.
        :param normalize:  If True, will normalize the entropy between 0 and 1.
        :return: A list of entropy values for the running window, None on error.
        """
        if window < 2:
            raise ValueError

        if len(inputbytes) < window:
            return None

        # Capture the running time
        start_time = time.time()

        # The counts
        bytescounter = Counter()
        # The current window for calculations
        byteswindow = deque()
        # The data to calculate
        inputbytesqueue = deque(inputbytes)

        entropy_list = list()

        # Initially fill up the window
        for i in range(0, window):
            currchar = inputbytesqueue.popleft()
            byteswindow.append(currchar)
            bytescounter[currchar] += 1

        # Calculate the initial entropy
        current_entropy = float(0)
        for b in bytescounter:
            if bytescounter[b] > 0:
                current_entropy -= ((float(bytescounter[b]) / window) *
                                    math.log2(float(bytescounter[b]) /
                                              window))

        # Add entropy value to output
        entropy_list.append(current_entropy)

        while True:
            # If there is no more input, break
            try:
                currchar = inputbytesqueue.popleft()
            except IndexError:
                break

            # Remove the old byte from current window and calculate
            oldchar = byteswindow.popleft()
            current_entropy += ((float(bytescounter[oldchar]) / window) *
                               math.log2(float(bytescounter[oldchar]) /
                                         window))
            bytescounter[oldchar] -= 1
            if bytescounter[oldchar] > 0:
                current_entropy -= ((float(bytescounter[oldchar]) / window)
                                    * math.log2(float(bytescounter[oldchar]) /
                                                window))

            # Calculate the newest added byte to the window
            if bytescounter[currchar] > 0:
                current_entropy += ((float(bytescounter[currchar]) /
                                     window) *
                                    math.log2(float(bytescounter[currchar]) /
                                              window))
            byteswindow.append(currchar)
            bytescounter[currchar] += 1
            current_entropy -= ((float(bytescounter[currchar]) / window) *
                                math.log2(float(bytescounter[currchar]) /
                                          window))

            # Add entropy value to output
            entropy_list.append(current_entropy)

        # Normalize if desired
        self.normalize[window] = normalize
        if normalize is True:
            self.entropy_data[window] = [i/8 for i in entropy_list]
        else:
            self.entropy_data[window] = entropy_list

        end_time = time.time()
        self.calctime[window] = end_time - start_time

        # Return the data
        return self.entropy_data[window]

    def read(self, sqlite_filename):
        """
        Reads the running window entropy from a Sqlite DB file.

        Deprecated.  Keeping to reuse logic later.

        :param sqlite_filename: SQLite file name
        :return: A dict containing the running window entropy values, keyed
            by running window size.
        """
        conn = sqlite3.connect(sqlite_filename)
        cursor = conn.cursor()

        results = None

        # Get metadata...
        try:
            cursor.execute('SELECT * FROM metadata;')
            results = cursor.fetchone()
        except sqlite3.OperationalError:
            # Oh well, the table doesn't exist...
            pass

        # Store the metadata info if it is available...
        if results:
            self.metadata['filepath'] = results[1]
            self.metadata['filesize'] = results[2]
            self.metadata['filetype'] = results[3]
            self.metadata['fileentropy'] = results[4]
            self.metadata['md5'] = results [5]
            self.metadata['sha256'] = results[6]
            self.metadata['dbfile'] = results[7]

        # Find the windows calculated...
        cursor.execute('SELECT * FROM windows;')
        results = cursor.fetchone()
        windows = list()

        while results:
            windows.append(results[1])
            self.normalize[results[1]] = results[2]
            self.calctime[results[1]] = results[3]
            results = cursor.fetchone()

        # Setup running entropy variables...
        self.entropy_data = dict()
        for w in windows:
            self.entropy_data[w] = list()

        # Read in running entropy values...
        cursor.execute('SELECT * FROM windowentropy ORDER BY offset;')
        results = cursor.fetchone()

        while results:
            self.entropy_data[results[1]].append(results[3])
            results = cursor.fetchone()

        conn.close()

        return self.entropy_data

    def write(self, sqlite_filename, metadata=None):
        """
        Writes entropy data to sqlite file.

        Deprecated.  Keeping to reuse logic later.

        :param sqlite_filename: SQLite file name
        :param metadata: Metadata dict to write to the DB file.  Only use this
            if you know what you are doing.
        :return: Nothing
        """
        # Create malware table structure...
        conn = sqlite3.connect(sqlite_filename)
        cursor = conn.cursor()

        # Save the metadata for later...
        self.metadata = metadata

        # If we are given metadata, store it...
        if metadata:
            # Create table for metadata in malware db...
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS metadata(' +
                'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                'filepath TEXT NOT NULL,'
                'filesize INT NOT NULL,'
                'filetype TEXT NOT NULL,'
                'fileentropy REAL NOT NULL,'
                'MD5 TEXT NOT NULL,'
                'SHA256 TEXT NOT NULL,'
                'DBFile TEXT NOT NULL'
            ');')
            conn.commit()
            # Prepare and execute SQL for the metadata...
            sql = "INSERT INTO metadata (filepath, filesize, filetype, " + \
                  "fileentropy, MD5, SHA256, DBFile) VALUES " + \
                  "(:filepath, :filesize, :filetype, :fileentropy, " + \
                  ":md5, :sha256, :dbfile);"
            params = {'filepath': metadata.get('filepath', ''),
                      'filesize': metadata.get('filesize', ''),
                      'filetype': metadata.get('filetype', ''),
                      'fileentropy': metadata.get('fileentropy', ''),
                      'md5': metadata.get('md5', ''),
                      'sha256': metadata.get('sha256', ''),
                      'dbfile': metadata.get('dbfile', '')}
            cursor.execute(sql, params)
            conn.commit()

        # Create table for window entropy in malware db...
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS windowentropy(' +
            'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
            'windowsize INT NOT NULL,'
            'offset INT NOT NULL,'
            'entropy REAL NOT NULL'
            ');')
        conn.commit()

        # Create table for window sizes in malware db...
        cursor.execute('CREATE TABLE IF NOT EXISTS windows(' +
                               'ID INTEGER PRIMARY KEY AUTOINCREMENT,'
                               'windowsize INT NOT NULL,'
                               'normalized INT NOT NULL,'
                               'calctime REAL NOT NULL'
                               ');')
        conn.commit()

        # Add running entropy to the database...
        for w in self.entropy_data:
            malware_offset = 0
            for r in self.entropy_data[w]:
                sql = "INSERT INTO windowentropy " + \
                      "(windowsize, offset, entropy) " + \
                      "VALUES (:windowsize, :offset, :entropy);"
                params = {'windowsize': w,
                          'offset': malware_offset,
                          'entropy': r}
                cursor.execute(sql, params)
                malware_offset += 1

            # Add the window size to the database signifying it is done...
            sql = "INSERT INTO windows (windowsize, normalized, calctime) " + \
                  "VALUES (:windowsize, :normalized, :calctime)"
            params = {'windowsize': w, 'normalized': self.normalize[w],
                      'calctime': self.calctime[w]}
            cursor.execute(sql, params)

        # Commit all our data...
        conn.commit()
        conn.close()

    def resample_rwe(self, window_size=256, number_of_data_points=1024):
        """
        Returns a resampled numpy array from the original running window
        entropy with a given window_size.

        :param window_size:  The size in bytes of the running window
        entropy window size.  This must have been calculated previously!
        :param number_of_data_points: The number of data points you want in
        your new data set.
        :return: xnew,ynew where xnew and ynew are numpy arrays that have
        been interpolated.
        """
        if window_size not in self.entropy_data:
            # Nothing to return if it does not exist.
            return None
        rwe = np.array(self.entropy_data[window_size])
        x_rwe = list(range(len(rwe)))
        step = len(x_rwe)/number_of_data_points
        xnew = np.arange(0, len(x_rwe), step)
        interp_rwe = interpolate.interp1d(x_rwe, rwe)
        ynew = interp_rwe(xnew)
        return xnew, ynew