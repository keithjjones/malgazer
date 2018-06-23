# Entropy Module
import math
import sqlite3
import os
import json
import time
from collections import Counter
from collections import deque
from scipy import interpolate
import pandas as pd
import numpy as np
from numba import jit


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
                current_entropy -= calculate_entropy(bytescounter[b], window)

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
            current_entropy += calculate_entropy(bytescounter[oldchar], window)
            bytescounter[oldchar] -= 1
            if bytescounter[oldchar] > 0:
                current_entropy -= calculate_entropy(bytescounter[oldchar], window)

            # Calculate the newest added byte to the window
            if bytescounter[currchar] > 0:
                current_entropy += calculate_entropy(bytescounter[currchar], window)
            byteswindow.append(currchar)
            bytescounter[currchar] += 1
            current_entropy -= calculate_entropy(bytescounter[currchar], window)

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
            return None, None
        rwe = np.array(self.entropy_data[window_size])
        x_rwe = list(range(len(rwe)))
        step = (len(x_rwe)-1)/number_of_data_points
        xnew = np.arange(0, len(x_rwe)-1, step)
        interp_rwe = interpolate.interp1d(x_rwe, rwe)
        ynew = interp_rwe(xnew)
        if len(ynew) != number_of_data_points:
            raise Exception('Unhandled Exception')
        return xnew, ynew


def resample_rwe(in_filename, out_filename, number_of_data_points=1024, compression='bz2'):
    """
    Resamples a RWE CSV file to the number_of_data_points.

    :param in_filename:  The filename of the raw data.
    :param out_filename:  The file name of the output resampled data.
    :param number_of_data_points: The number of data points you want in
    your new data set.
    :param compression:  The compression to use for the input and output CSVs.
    :return: Nothing
    """
    # Start the timer
    start_time = time.time()

    print("Resampling raw RWE: {0}".format(in_filename))
    chunksize = 10
    chunk_number = 0
    for in_df in pd.read_csv(in_filename, index_col=0, compression=compression, chunksize=chunksize):
        chunk_number += 1
        print("\tResampling rows, Chunk {0}".format(chunk_number))
        rows_to_add = []
        n_rows = len(in_df.index)
        row_number = 1
        for index, row in in_df.iterrows():
            rwe = np.array(row)
            x_rwe = list(range(len(rwe)))
            step = (len(x_rwe)-1)/number_of_data_points
            xnew = np.arange(0, len(x_rwe)-1, step)
            interp_rwe = interpolate.interp1d(x_rwe, rwe)
            ynew = interp_rwe(xnew)
            if len(ynew) != number_of_data_points:
                raise Exception('Unhandled Exception - This should not happen!')
            ds = pd.Series(ynew)
            ds.name = index
            rows_to_add.append(ds)
            if row_number % 100 == 0:
                print("\t\t{0} out of {1} total rows resampled in chunk {2}..."
                      .format(row_number, n_rows, chunk_number))
            row_number += 1
        print("\tAssembling new DataFrame for chunk {0}...".format(chunk_number))
        out_df = pd.DataFrame()
        out_df = out_df.append(rows_to_add)
        print("\tWriting resampled RWE chunk {0} to file: {1}"
              .format(chunk_number, out_filename))
        if chunk_number == 1:
            out_df.to_csv(out_filename, compression=compression)
        else:
            out_df.to_csv(out_filename, compression=compression, mode='a')
    print("Total elapsed time {0:.6f} seconds".format(round(time.time() - start_time, 6)))


def calculate_entropy(counts, windowsize):
    """
    Calculates entropy based upon counts and window size.

    :param counts:  The count of the symbol.
    :param windowsize:  The size of the window.
    :return:  The entropy as a float.
    """
    ent = ((float(counts) / windowsize) * math.log(float(counts) / windowsize, 2))
    return ent