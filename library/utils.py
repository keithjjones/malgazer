# Utilities class
import os
import re
import time
import pandas as pd
import numpy as np
from .files import FileObject
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

class Utils(object):
    def __init__(self):
        super(Utils, self).__init__()

    @staticmethod
    def batch_running_window_entropy(in_directory=None,
                                     out_directory=None,
                                     window_sizes=[256],
                                     normalize=True):
        """
        Calculates the running window entropy of a directory containing
        malware samples that is named from their SHA256 value.  It will
        skip all other files.

        :param in_directory:  The input directory for malware.
        :param out_directory: The output directory for calculated data.
        :param window_sizes: A list of window sizes to calculate.
        :param normalize: Set to false to not normalize.
        :return: Nothing
        """
        if in_directory is None or out_directory is None:
            raise ValueError('Input and output directories must be real.')
        if len(window_sizes) < 1:
            raise ValueError('Specify a window size in a list.')

        # Test to make sure the input directory exists, will throw exception
        # if it does not exist.
        os.stat(in_directory)

        # Start the timer
        start_time = time.time()

        # The RE for malware files with sha256 as the name.
        malware_files_re = re.compile('[a-z0-9]{64}',
                                      flags=re.IGNORECASE)
        samples_processed = 0
        for root, dirs, files in os.walk(in_directory):
            for file in files:
                if malware_files_re.match(file):
                    # Start the timer
                    start_load_time = time.time()

                    print("Input file: {0}".format(file))
                    subdir = root[len(in_directory):]

                    # Create the malware file name...
                    malwarepath = os.path.join(root, file)
                    try:
                        m = FileObject(malwarepath)
                    except:
                        continue

                    print("\tCalculating: {0} Type: {1}".format(m.malware.filename, m.malware.filetype))

                    # Create the DB file name...
                    datadir = os.path.join(out_directory, subdir)
                    picklefile = os.path.join(datadir, file) + ".pickle.gz"

                    print("\tSaving data to {0}".format(picklefile))

                    # Create the directory if needed...
                    try:
                        os.stat(datadir)
                    except:
                        os.makedirs(datadir)

                    # Remove old pickle files...
                    if os.path.exists(picklefile):
                        os.remove(picklefile)

                    # Calculate the entropy of the file...
                    fileentropy = m.entropy(normalize)

                    # Calculate the window entropy for malware samples...
                    if window_sizes is not None:
                        # Iterate through the window sizes...
                        for w in window_sizes:
                            if w < m.malware.file_size:
                                print("\t\tCalculating window size {0:,}".format(w))

                                # Calculate running entropy...
                                rwe = m.running_entropy(w, normalize)

                        # Write the running entropy...
                        m.write(picklefile)

                    print("\tElapsed time {0:.6f} seconds".format(round(time.time() - start_load_time, 6)))

                    samples_processed += 1
                    print("{0:n} samples processed...".format(samples_processed))
        print("\tTotal elapsed time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
        print("{0:n} total samples processed...".format(samples_processed))

    @staticmethod
    def batch_tsfresh_rwe_data(in_directory=None,
                               datapoints=512,
                               window_size=256):
        """
        Return extracted features of malware using tsfresh.

        :param in_directory:  The directory containing the malware pickle files
        created in with the batch function above.
        :param datapoints: The number of datapoints to resample RWE.
        :param window_size:  The window size of the RWE, that must be already
        calculated.
        :return:  A Pandas dataframe containing the tsfresh features.
        """
        # Start the timer
        start_time = time.time()
        # Check to see that the input directory exists, this will throw an
        # exception if it does not exist.
        os.stat(in_directory)
        # Only find pickle malware files created by the batch function above.
        malware_files_re = re.compile('[a-z0-9]{64}.pickle.gz',
                                      flags=re.IGNORECASE)
        df = pd.DataFrame(columns=['id', 'offset', 'rwe'])
        samples_processed = 0
        for root, dirs, files in os.walk(in_directory):
            for file in files:
                if malware_files_re.match(file):
                    start_load_time = time.time()
                    print("Reading file: {0}".format(file))
                    f = FileObject.read(os.path.join(root, file))
                    running_entropy = f.malware.runningentropy
                    if window_size in running_entropy.entropy_data:
                        # Reduce RWE data points
                        xnew, ynew = running_entropy.resample_rwe(window_size=window_size,
                                                                  number_of_data_points=datapoints)
                        # Create dataframe
                        d = pd.DataFrame(columns=['id', 'offset', 'rwe'])
                        d['rwe'] = ynew
                        d['id'] = f.malware.sha256.upper()
                        d['offset'] = np.arange(0, datapoints)
                        df = df.append(d, ignore_index=True)
                    else:
                        print("ERROR: Window size {0} not in this pickle file!".format(window_size))
                    print("\tElapsed time {0:.6f} seconds".format(round(time.time() - start_load_time, 6)))
                    samples_processed += 1
                    print("{0:n} samples processed...".format(samples_processed))
        print("Calculating TSFresh Features...")
        start_tsfresh_time = time.time()
        settings = EfficientFCParameters()
        extracted_features = extract_features(df, column_id="id",
                                              column_sort='offset',
                                              default_fc_parameters=settings)
        print("\tElapsed time {0:.6f} seconds".format(
            round(time.time() - start_tsfresh_time, 6)))
        print("\tTotal elapsed time {0:.6f} seconds".format(
            round(time.time() - start_time, 6)))
        print("{0:n} total samples processed...".format(samples_processed))
        return extracted_features
