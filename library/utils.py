# Utilities Module
import os
import re
import time
import shutil
import pickle
import gzip
import csv
import pandas as pd
import numpy as np
from .files import FileObject
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
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

        print("Starting running window entropy batch processing for malware samples...")

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
        print("Total elapsed time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
        print("{0:n} total samples processed...".format(samples_processed))

    @staticmethod
    def batch_preprocess_rwe_data(in_directory = None,
                                  datapoints = 512,
                                  window_size = 256):
        """
        Return rwe of malware in a dataframe.

        :param in_directory:  The directory containing the malware pickle files
        created in with the batch function above.
        :param datapoints: The number of datapoints to resample RWE.
        :param window_size:  The window size of the RWE, that must be already
        calculated.
        :return:  A Pandas dataframe containing the rwe.
        """
        print("Starting batch processing of running window entropy for malware samples...")
        # Keep track so we don't duplicate work
        processed_sha256 = []

        # Start the timer
        start_time = time.time()
        # Check to see that the input directory exists, this will throw an
        # exception if it does not exist.
        os.stat(in_directory)
        # Only find pickle malware files created by the batch function above.
        malware_files_re = re.compile('[a-z0-9]{64}.pickle.gz',
                                      flags=re.IGNORECASE)
        df = pd.DataFrame()
        samples_processed = 0
        for root, dirs, files in os.walk(in_directory):
            for file in files:
                if malware_files_re.match(file):
                    start_load_time = time.time()
                    print("Reading file: {0}".format(file))
                    f = FileObject.read(os.path.join(root, file))
                    if f.malware.sha256.upper() not in processed_sha256:
                        running_entropy = f.malware.runningentropy

                        if window_size in running_entropy.entropy_data:
                            # Reduce RWE data points
                            xnew, ynew = running_entropy.resample_rwe(window_size=window_size,
                                                                      number_of_data_points=datapoints)
                            s = pd.Series(ynew)
                            s.name = f.malware.sha256.upper()
                            df = df.append(s)
                            processed_sha256.append(f.malware.sha256.upper())
                            print("\tElapsed time {0:.6f} seconds".format(
                                 round(time.time() - start_load_time, 6)))
                            samples_processed += 1
                            print("{0:n} samples processed...".format(
                                 samples_processed))
                        else:
                            print(
                                "ERROR: Window size {0} not in this pickle file!".format(window_size))
        print("Total elapsed time {0:.6f} seconds".format(
              round(time.time() - start_time, 6)))
        print("{0:n} total samples processed...".format(samples_processed))
        return df

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
        :return:  A Pandas dataframe containing the tsfresh features, and the
        raw data frame as a tuple.
        """
        print("Starting batch processing of tsfresh on running window entropy for malware samples...")

        # Keep track so we don't duplicate work
        processed_sha256 = []

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
                    if f.malware.sha256.upper() not in processed_sha256:
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
                            processed_sha256.append(f.malware.sha256.upper())
                            print("\tElapsed time {0:.6f} seconds".format(
                            round(time.time() - start_load_time, 6)))
                            samples_processed += 1
                            print("{0:n} samples processed...".format(samples_processed))
                        else:
                            print("ERROR: Window size {0} not in this pickle file!".format(window_size))
        print("Calculating TSFresh Features...")
        start_tsfresh_time = time.time()
        settings = EfficientFCParameters()
        extracted_features = extract_features(df,
                                              column_id="id",
                                              column_sort='offset',
                                              default_fc_parameters=settings,
                                              impute_function=impute)
        print("\tElapsed time {0:.6f} seconds".format(
            round(time.time() - start_tsfresh_time, 6)))
        print("Total elapsed time {0:.6f} seconds".format(
            round(time.time() - start_time, 6)))
        print("{0:n} total samples processed...".format(samples_processed))
        return extracted_features, df

    @staticmethod
    def extract_tsfresh_relevant_features(extracted_features, classifications):
        """
        Return only relevant features.

        :param extracted_features:  A dataframe from the tsfresh rwe function
        above.
        :param classifications:  A list of ordered classifications for the features.
        :return:  A DataFrame of relevant features.
        """
        impute(extracted_features)
        features_filtered = select_features(extracted_features, classifications)
        return features_filtered

    @staticmethod
    def get_classifications_from_path(in_directory=None):
        """
        Loads classifications from key words in the path.

        :param in_directory:  This is the directory containing batch processed
        samples with the batch function above (results are pickled).
        :return: A DataFrame with an index of the sha256 and the value of
        the classification guessed from the full path name.
        """
        print("Starting classifications from path for malware samples...")

        # Keep track so we don't duplicate work
        processed_sha256 = []

        df = pd.DataFrame(columns=['classification'])

        # Check to see that the input directory exists, this will throw an
        # exception if it does not exist.
        os.stat(in_directory)
        # The RE for malware files with sha256 as the name.
        malware_files_re = re.compile('[a-z0-9]{64}',
                                      flags=re.IGNORECASE)
        samples_processed = 0
        for root, dirs, files in os.walk(in_directory):
            for file in files:
                if malware_files_re.match(file):
                    samples_processed += 1

                    f = FileObject.read(os.path.join(root, file))

                    if f.malware.sha256.upper() not in processed_sha256:
                        classified = ""
                        if "encrypted" in root.lower():
                            classified = "Encrypted"
                        elif "unpacked" in root.lower():
                            classified = "Unpacked"
                        elif "packed" in root.lower():
                            classified = "Packed"
                        else:
                            classified = ""

                        if "malware" in root.lower():
                            classified += "-Malware"
                        elif "pup" in root.lower():
                            classified += "-PUP"
                        elif "trusted" in root.lower():
                            classified += "-Trusted"
                        else:
                            classified += ""

                        d = dict()
                        d['classification'] = classified
                        ds = pd.Series(d)
                        ds.name = f.malware.sha256.upper()
                        df = df.append(ds)

                        processed_sha256.append(f.malware.sha256.upper())

                        samples_processed += 1
        return df


    @staticmethod
    def parse_classifications_from_path(classifications):
        """
        Parses the classifications from guessed classifications from the paths.

        :param classifications:  A dataframe holding the guessed classifications.
        :return:  The parsed classifications, as a DataFrame.
        """
        cls = pd.DataFrame(columns=['classification'])
        for index, row in classifications.iterrows():
            cl = row[0]
            c = cl.split('-')
            d = dict()
            d['classification'] = c[0]
            ds = pd.Series(d)
            ds.name = index
            cls = cls.append(ds)
        return cls

    @staticmethod
    def save_processed_tsfresh_data(raw_data,
                                    classifications,
                                    extracted_features,
                                    datadir):
        """
        Saves the pieces of data that were calculated with the functions above
        to a data directory.

        :param raw_data:  The raw data loaded from the malware sample set.
        :param classifications:  A DataFrame with index of sha256 and
        value of classifications.
        :param extracted_features:  The TSFresh extracted features data.
        :param datadir: The data directory to store the data.  Any old data
        will be deleted!
        :return:  Nothing
        """
        print("Starting the data save for the tsfresh data for malware samples...")
        # Remove previous data
        try:
            shutil.rmtree(datadir)
        except:
            pass
        os.makedirs(datadir)
        # Raw data
        raw_data.to_csv(os.path.join(datadir, "raw_data.csv.gz"), compression='gzip')
        with gzip.open(os.path.join(datadir, "raw_data.pickle.gz"), 'wb') as file:
            pickle.dump(raw_data, file)
        # Classifications
        classifications.to_csv(os.path.join(datadir, "classifications.csv.gz"), compression='gzip')
        with gzip.open(os.path.join(datadir, "classifications.pickle.gz"), 'wb') as file:
            pickle.dump(classifications, file)
        # Extracted Features
        extracted_features.to_csv(os.path.join(datadir, 'extracted_features.csv.gz'),
                                  compression='gzip')
        with gzip.open(os.path.join(datadir, "extracted_features.pickle.gz"), 'wb') as file:
            pickle.dump(extracted_features, file)

    @staticmethod
    def load_processed_tsfresh_data(datadir):
        """
        Loads the data saved from tsfresh.

        :param datadir:  The data directory that contains the data.
        :return:  raw_data, classifications, extracted_features tuple
        """
        # Check to see that the data directory exists, this will throw an
        # exception if it does not exist.
        os.stat(datadir)
        with gzip.open(os.path.join(datadir, "raw_data.pickle.gz"), 'rb') as file:
            df = pickle.load(file)
        with gzip.open(os.path.join(datadir, "classifications.pickle.gz"), 'rb') as file:
            classifications = pickle.load(file)
        with gzip.open(os.path.join(datadir, "extracted_features.pickle.gz"), 'rb') as file:
            extracted_features = pickle.load(file)
        return df, classifications, extracted_features

    @staticmethod
    def save_preprocessed_data(raw_data,
                               classifications,
                               datadir):
        """
        Saves the pieces of data that were calculated with the functions above
        to a data directory.

        :param raw_data:  The raw data loaded from the malware sample set.
        :param classifications:  A DataFrame with index of sha256 and
        value of classifications.
        :param datadir: The data directory to store the data.  Any old data
        will be deleted!
        :return:  Nothing
        """
        print("Starting the data save for the preprocessed data for malware samples...")
        # Remove previous data
        try:
            shutil.rmtree(datadir)
        except:
            pass
        os.makedirs(datadir)
        # Raw data
        raw_data.to_csv(os.path.join(datadir, "raw_data.csv.gz"), compression='gzip')
        with gzip.open(os.path.join(datadir, "raw_data.pickle.gz"), 'wb') as file:
            pickle.dump(raw_data, file)
        # Classifications
        classifications.to_csv(os.path.join(datadir, "classifications.csv.gz"), compression='gzip')
        with gzip.open(os.path.join(datadir, "classifications.pickle.gz"), 'wb') as file:
            pickle.dump(classifications, file)

    @staticmethod
    def load_preprocessed_data(datadir):
        """
        Loads the data saved from preprocessing.

        :param datadir:  The data directory that contains the data.
        :return:  raw_data, classifications tuple
        """
        # Check to see that the data directory exists, this will throw an
        # exception if it does not exist.
        os.stat(datadir)
        with gzip.open(os.path.join(datadir, "raw_data.pickle.gz"), 'rb') as file:
            df = pickle.load(file)
        with gzip.open(os.path.join(datadir, "classifications.pickle.gz"), 'rb') as file:
            classifications = pickle.load(file)
        return df, classifications
