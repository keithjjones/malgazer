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
from .files import Sample
from .entropy import resample
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import threading
from time import sleep
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


class Utils(object):

    translate_classifications = {
        'Trj': 'Trojan',
        'PUP': 'PUA',
        'Adw': 'Adware',
        'Drp': 'Dropper',
        'Wrm': 'Worm',
        'Bkdr': 'Backdoor',
        'Cryp': 'Ransom',
        'PUA': 'PUA',
        'RiskTool': 'PUA',
        'Generic': 'Generic',
        'VirTool': 'Virus',
        'VirLock': 'Ransom',
        'HackTool': 'PUA',
        'RemoteAdmin': 'PUA',
        'Trojan': 'Trojan',
        'Mal': 'Generic',
        'Malware': 'Generic',
        'Worm': 'Worm',
        'Troj': 'Trojan',
        'FakeAV': 'FakeAV',
        'DwnLdr': 'Downloader',
        'TrojanDownloader': 'Downloader',
    }

    def __init__(self):
        super(Utils, self).__init__()

    @staticmethod
    def _calculate_rwe(filename, window_size=256, normalize=True,
                       number_of_data_points=[1024]):
        """
        Internal method to calculate the RWE for a sample.

        :param filename:  The file name of the sample to calculate.
        :param window_size: A window size to calculate.
        :param normalize: Set to False to not normalize.
        :param number_of_data_points: The number of data points you want in
        your new data set.  The output will be resampled to this many data points.
        The number of data points is specified in a list.
        :return:  Returns a pandas data series so it can be added to a
        data frame.  The output is a dict with the key of the datapoints.
        """
        s = Sample()
        s.fromfile(filename)
        rwe = s.running_window_entropy(window_size, normalize)
        output_dict = {}
        for datapoints in number_of_data_points:
            ds = pd.Series(resample(rwe, datapoints))
            ds.name = s.sha256
            output_dict[datapoints] = ds
        return output_dict

    @staticmethod
    def extract_rwe_features_from_directory(in_directory=None,
                                            out_directory=None,
                                            window_size=256,
                                            normalize=True,
                                            number_of_data_points=[1024],
                                            njobs=os.cpu_count(),
                                            batch_size=1000):
        """
        Calculates the running window entropy of a directory containing
        malware samples that are named from their SHA256 value.  It will
        skip all other files.  The output format will be HDF.

        :param in_directory:  The input directory for malware.
        :param out_directory: The output directory for calculated data.
        :param window_size: A window size to calculate.
        :param normalize: Set to False to not normalize.
        :param number_of_data_points: The number of data points you want in
        your new data set.  The output will be resampled to this many data points.
        The number of data points is specified in a list.
        :param njobs: The number of processes to use.
        :param batch_size:  The number of rows to write to the HDF file in each chunk.
        :return: Nothing
        """
        if in_directory is None or out_directory is None:
            raise ValueError('Input and output directories must be real.')
        if isinstance(number_of_data_points, int):
            number_of_data_points = list(number_of_data_points)
        if not isinstance(number_of_data_points, list):
            raise ValueError('Specify number of datapoints size in a list.')
        if njobs < 1:
            raise ValueError('The number of jobs needs to be >= 1')

        print("Starting running window entropy feature extractor for malware samples in {0}".format(in_directory))

        # Test to make sure the input directory exists, will throw exception
        # if it does not exist.
        os.stat(in_directory)

        # Start the timer
        start_time = time.time()

        # The RE for malware files with sha256 as the name.
        malware_files_re = re.compile('[a-z0-9]{64}',
                                      flags=re.IGNORECASE)
        samples_processed = 0
        saved_futures = {}
        rows_to_add = {n: list() for n in number_of_data_points}
        hdffilenames = {n: os.path.join(out_directory, 'rwe_window_{0}_datapoints_{1}.hdf'.format(window_size, n)) for n in number_of_data_points}
        for datapoint in hdffilenames:
            if os.path.isfile(hdffilenames[datapoint]):
                os.remove(hdffilenames[datapoint])
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            try:
                for root, dirs, files in os.walk(in_directory):
                    for file in files:
                        filename = os.path.join(root, file)
                        if malware_files_re.match(file):
                            future = executor.submit(Utils._calculate_rwe,
                                                     filename, window_size,
                                                     normalize,
                                                     number_of_data_points)
                            saved_futures[future] = filename
                for future in as_completed(saved_futures):
                    samples_processed += 1
                    result = future.result()
                    for datapoint in result:
                        rows_to_add[datapoint].append(result[datapoint].copy())
                    print("Processed file: {0}".format(saved_futures[future]))
                    print("\t{0:,} samples processed...".format(samples_processed))
                    if samples_processed % batch_size == 0:
                        for datapoint in rows_to_add:
                            print("Writing chunk to HDF: {0}".format(hdffilenames[datapoint]))
                            print("\tAssembling DataFrame...")
                            df = pd.DataFrame(rows_to_add[datapoint])
                            print("\tWriting HDF...")
                            df.to_hdf(hdffilenames[datapoint], 'rwe', mode='a', append=True, format='table')
                            print("\tClear rows to add...")
                        rows_to_add = {n: list() for n in number_of_data_points}
                for datapoint in rows_to_add:
                    if len(rows_to_add[datapoint]) > 0:
                        print("Writing last chunk to HDF: {0}".format(hdffilenames[datapoint]))
                        print("\tAssembling DataFrame...")
                        df = pd.DataFrame(rows_to_add[datapoint])
                        print("\tWriting HDF...")
                        df.to_hdf(hdffilenames[datapoint], 'rwe', mode='a', append=True, format='table')
            except KeyboardInterrupt:
                print("Shutting down gracefully...")
                executor.shutdown(wait=False)
        print("Total elapsed time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
        print("{0:,} total samples processed...".format(samples_processed))

    @staticmethod
    def _calculate_gist(filename):
        """
        Internal method to calculate the GIST for a sample.

        :param filename:  The file name of the sample to calculate.
        :return:  Returns a pandas data series so it can be added to a
        data frame.
        """
        s = Sample()
        s.fromfile(filename)
        gist = s.gist_data
        ds = pd.Series(gist)
        ds.name = s.sha256
        return ds

    @staticmethod
    def extract_gist_features_from_directory(in_directory=None,
                                             out_directory=None,
                                             njobs=os.cpu_count(),
                                             batch_size=1000):
        """
        Calculates the GIST of a directory containing
        malware samples that are named from their SHA256 value.  It will
        skip all other files.  The output format will be HDF.

        :param in_directory:  The input directory for malware.
        :param out_directory: The output directory for calculated data.
        :param njobs: The number of processes to use.
        :param batch_size:  The number of rows to write to the HDF file in each chunk.
        :return: Nothing
        """
        if in_directory is None or out_directory is None:
            raise ValueError('Input and output directories must be real.')
        if njobs < 1:
            raise ValueError('The number of jobs needs to be >= 1')

        print("Starting GIST feature extractor for malware samples in {0}".format(in_directory))

        # Test to make sure the input directory exists, will throw exception
        # if it does not exist.
        os.stat(in_directory)

        # Start the timer
        start_time = time.time()

        # The RE for malware files with sha256 as the name.
        malware_files_re = re.compile('[a-z0-9]{64}',
                                      flags=re.IGNORECASE)
        samples_processed = 0
        saved_futures = {}
        rows_to_add = []
        hdffilename = os.path.join(out_directory, 'gist.hdf')
        if os.path.isfile(hdffilename):
            os.remove(hdffilename)
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            try:
                for root, dirs, files in os.walk(in_directory):
                    for file in files:
                        filename = os.path.join(root, file)
                        if malware_files_re.match(file):
                            future = executor.submit(Utils._calculate_gist, filename)
                            saved_futures[future] = filename
                for future in as_completed(saved_futures):
                    samples_processed += 1
                    result = future.result()
                    rows_to_add.append(result.copy())
                    print("Processed file: {0}".format(saved_futures[future]))
                    print("\t{0:,} samples processed...".format(samples_processed))
                    if samples_processed % batch_size == 0:
                        print("Writing chunk to HDF: {0}".format(hdffilename))
                        print("\tAssembling DataFrame...")
                        df = pd.DataFrame(rows_to_add)
                        print("\tWriting HDF...")
                        df.to_hdf(hdffilename, 'gist', mode='a', append=True, format='table')
                        print("\tClear rows to add...")
                        rows_to_add = []
                if len(rows_to_add) > 0:
                    print("Writing last chunk to HDF: {0}".format(hdffilename))
                    print("\tAssembling DataFrame...")
                    df = pd.DataFrame(rows_to_add)
                    print("\tWriting HDF...")
                    df.to_hdf(hdffilename, 'gist', mode='a', append=True, format='table')
            except KeyboardInterrupt:
                print("Shutting down gracefully...")
                executor.shutdown(wait=False)
        print("Total elapsed time {0:.6f} seconds".format(round(time.time() - start_time, 6)))
        print("{0:,} total samples processed...".format(samples_processed))

    @staticmethod
    def estimate_vt_classifications_from_csv(filename, products=None):
        """
        Reads a CSV containing VT classifications and attempts to estimate
        the accurate classification from several AV vendors.

        :param filename:  The file name of the CSV containing the VT data.
        :param products:  The list of products (columns) to use in the classification.
        The order is important.  Products earlier in the list take precedence over
        products later in the list.
        :return:  A DataFrame with the classification for each hash.
        """
        df = pd.read_csv(filename, index_col=0)
        return Utils.estimate_vt_classifications_from_DataFrame(df, products)

    @staticmethod
    def estimate_vt_classifications_from_hdf(filename, products=None):
        """
        Reads a HDF file containing VT classifications and attempts to estimate
        the accurate classification from several AV vendors.

        :param filename:  The file name of the HDF containing the VT data.
        :param products:  The list of products (columns) to use in the classification.
        The order is important.  Products earlier in the list take precedence over
        products later in the list.
        :return:  A DataFrame with the classification for each hash.
        """
        df = pd.read_hdf(filename, 'data')
        return Utils.estimate_vt_classifications_from_DataFrame(df, products)

    @staticmethod
    def estimate_vt_classifications_from_DataFrame(classifications, products=None):
        """
        Reads a DataFrame containing VT classifications and attempts to estimate
        the accurate classification from several AV vendors.

        :param classifications:  The file name of the CSV containing the VT data.
        :param products:  The list of products (columns) to use in the classification.
        The order is important.  Products earlier in the list take precedence over
        products later in the list.
        :return:  A DataFrame with the classification for each hash.
        """
        if products is None:
            products = ['Microsoft', 'Symantec', 'Kaspersky',
                        'Sophos', 'TrendMicro', 'ClamAV']

        classifications_prods = dict()

        for product in products:
            classifications_prods[product] = Utils.parse_vt_classifications(classifications,
                                                                            product)

        out_df = pd.DataFrame(columns=['classification'])

        translate_classifications = {k.lower(): v for k, v
                                         in Utils.translate_classifications.items()}

        for index, row in classifications.iterrows():
            current_classification = None
            for product in products:
                class_df = classifications_prods[product]
                c = class_df.loc[index.upper()]
                c = c['classification']
                if (c is not None
                        and c.lower() != 'none'
                        and c.strip() != ''
                        and current_classification is None):
                    current_classification = c
            if current_classification is not None:
                for string in translate_classifications:
                    if string in current_classification.lower():
                        current_classification = translate_classifications[string]
                        break
                d = dict()
                d['classification'] = current_classification
                ds = pd.Series(d)
                ds.name = index.upper()
                out_df = out_df.append(ds)
        return out_df

    @staticmethod
    def parse_vt_classifications(classifications, column_name):
        """
        Parses the classifications from a VT data set

        :param classifications:  The VT data set as a DataFrame.
        :param column_name:  Which column to parse in the DataFrame.
        :return: A DataFrame with the classifications, None if error.
        """
        cls = pd.DataFrame(columns=['classification'])
        for index, row in classifications.iterrows():
            if index.upper() not in cls.index:
                if column_name == 'Microsoft':
                    c = Utils.parse_microsoft_classification(row['Microsoft'])
                elif column_name == "Symantec":
                    c = Utils.parse_symantec_classification(row['Symantec'])
                elif column_name == "Kaspersky":
                    c = Utils.parse_kaspersky_classification(row['Kaspersky'])
                elif column_name == "Sophos":
                    c = Utils.parse_sophos_classification(row['Sophos'])
                elif column_name == "TrendMicro":
                    c = Utils.parse_trendmicro_classification(row['TrendMicro'])
                elif column_name == "K7AntiVirus":
                    c = Utils.parse_k7antivirus_classification(row['K7AntiVirus'])
                elif column_name == "ClamAV":
                    c = Utils.parse_clamav_classification(row['ClamAV'])
                elif column_name == "F-Prot":
                    c = Utils.parse_fprot_classification(row['F-Prot'])
                elif column_name == "Avast":
                    c = Utils.parse_avast_classification(row['Avast'])
                elif column_name == "Ikarus":
                    c = Utils.parse_ikarus_classification(row['Ikarus'])
                elif column_name == "Jiangmin":
                    c = Utils.parse_jiangmin_classification(row['Jiangmin'])
                elif column_name == "Emsisoft":
                    c = Utils.parse_generic_classification(row['Emsisoft'])
                elif column_name == "BitDefender":
                    c = Utils.parse_generic_classification(row['BitDefender'])
                elif column_name == "Arcabit":
                    c = Utils.parse_generic_classification(row['Arcabit'])
                elif column_name == "Ad-Aware":
                    c = Utils.parse_generic_classification(row['Ad-Aware'])
                elif column_name == "AVware":
                    c = Utils.parse_generic_classification(row['AVware'])
                elif column_name == "ALYac":
                    c = Utils.parse_generic_classification(row['ALYac'])
                elif column_name == "AVG":
                    c = Utils.parse_avg_classification(row['AVG'])
                elif column_name == "ZoneAlarm":
                    c = Utils.parse_generic_classification(row['ZoneAlarm'])
                elif column_name == "Zillya":
                    c = Utils.parse_generic_classification(row['Zillya'])
                elif column_name == "Yandex":
                    c = Utils.parse_generic_classification(row['Yandex'])
                elif column_name == "ViRobot":
                    c = Utils.parse_generic_classification(row['ViRobot'])
                elif column_name == "VIPRE":
                    c = Utils.parse_generic_classification(row['VIPRE'])
                elif column_name == "VBA32":
                    c = Utils.parse_generic_classification(row['VBA32'])
                elif column_name == "Webroot":
                    c = Utils.parse_webroot_classification(row['Webroot'])
                else:
                    # This is an error, so return None
                    return None
                if c is not None and c.lower() == 'misleading':
                    # This takes care of not so good signatures, such as Microsoft
                    c = None
                d = dict()
                d['classification'] = c
                ds = pd.Series(d)
                ds.name = index.upper()
                cls = cls.append(ds)
        return cls

    @staticmethod
    def parse_microsoft_classification(classification):
        """
        Parses the classification from a Microsoft VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split(':')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_symantec_classification(classification):
        """
        Parses the classification from a Symantec VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                if c is not None and c[0].lower() != 'smg':
                    if c[0] != 'W32':
                        c = c[0].split(' ')
                        c = c[0].split('!')
                        c = c[0].split('@')
                        return c[0]
                    else:
                        c = c[1].split('!')
                        c = c[0].split('@')
                        return c[0]
                else:
                    # This is Symantec Messaging Gateway string...
                    c = c[1]
                    c = c.split('!')
                    if c[1].lower() == 'gen':
                        return 'Generic'
                    else:
                        return c[1]
            except:
                pass
        return None

    @staticmethod
    def parse_kaspersky_classification(classification):
        """
        Parses the classification from a Kaspersky VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                c = c[0].split(':')
                if len(c) > 1:
                    if len(c) > 2 and c[1].lower() == 'heur':
                        c = c[2]
                        c = c.split('.')
                        return c[0]
                    else:
                        return c[1]
                else:
                    return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_sophos_classification(classification):
        """
        Parses the classification from a Sophos VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                if "(pua)" in classification.lower():
                    return classification
                else:
                    c = classification.split('/')
                    if len(c) > 1 and 'behav' in c[1].lower():
                        return c[0]
                    else:
                        return c[1]
            except:
                pass
        return None

    @staticmethod
    def parse_trendmicro_classification(classification):
        """
        Parses the classification from a TrendMicro VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('_')
                if c[0].lower() == 'pe' and len(c) > 1:
                    c = c[1]
                    c = c.split('.')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_k7antivirus_classification(classification):
        """
        Parses the classification from a K7AntiVirus VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split(' ')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_clamav_classification(classification):
        """
        Parses the classification from a ClamAV VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                return c[1]
            except:
                pass
        return None

    @staticmethod
    def parse_avast_classification(classification):
        """
        Parses the classification from a Avast VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split(':')
                c = c[1].split(' ')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_fprot_classification(classification):
        """
        Parses the classification from an F-Prot VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('/')
                c = c[1].split('.')
                c = c[0].split('!')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_ikarus_classification(classification):
        """
        Parses the classification from an Ikarus VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_jiangmin_classification(classification):
        """
        Parses the classification from a Jiangmin VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_webroot_classification(classification):
        """
        Parses the classification from a Webroot VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                return c[1]
            except:
                pass
        return None

    @staticmethod
    def parse_avg_classification(classification):
        """
        Parses the classification from an AVG VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                class_re = re.compile('.*\[(.*)\].*',
                                       flags=re.IGNORECASE)
                m = class_re.match(classification)
                if m:
                    c = m.group(1)
                    return c
                else:
                    c = classification.split(':')
                    c = c[1]
                    c = c.split(' ')
                    return c[0]
            except:
                pass
        return None

    @staticmethod
    def parse_generic_classification(classification):
        """
        Parses the classification from a generic VT string

        :param classification:  The VT string
        :return: The classification, or None if it could not parse.
        """
        if isinstance(classification, str):
            if classification == 'scan_error':
                return None
            try:
                c = classification.split('.')
                if c[0].lower() == 'gen:variant':
                    c = c[1]
                elif c[0].lower() == 'gen:win32':
                    c = c[1]
                elif c[0].lower() == 'application':
                    c = c[1]
                elif c[0].lower() == 'win32':
                    c = c[1]
                elif c[0].lower() == 'behaveslike':
                    c = c[2]
                else:
                    c = c[0]
                c = c.split(' ')[0]
                if ':' in c:
                    c = c.split(':')[1]
                return c
            except:
                pass
        return None

    @staticmethod
    def parse_classifications_with_delimiter(classifications,
                                             column_name,
                                             delimiter,
                                             split_index=1):
        """
        Parses the classification from a DataFrame where the format is:
        <CLASSIFICATION><DELIMITER><SUBCLASSIFICATION>...

        :param classifications:  The DataFrame containing classification info.
        The DF must have the index as the file hash.
        :param delimiter: the delimiter for the original classification info in
        the DataFrame.
        :param column_name: The column name containing the classification info.
        :param split_index:  The index after the split using the delimiter
        :return: A DataFrame of classifications for hashes.
        """
        cls = pd.DataFrame(columns=['classification'])
        for index, row in classifications.iterrows():
            if isinstance(row[column_name], str):
                cl = row[column_name]
                c = cl.split(delimiter)
                d = dict()
                d['classification'] = c[split_index]
                ds = pd.Series(d)
                ds.name = index.upper()
                cls = cls.append(ds)
        return cls

    @staticmethod
    def load_features(datadir, feature_type='rwe', filterhashes=False, *args, **kwargs):
        """
        Loads data from datadir, for feature_type, and sanity checks the data sets.

        :param datadir:  The data dir to load from.
        :param feature_type:   The feature type:  'rwe' or 'gist'.
        'rwe' requires a datapoints and windowsize named arguments.
        :param filterhashes:  Set to True to load hashes.csv and only show hashes that are inside that file.
        :return:  A 3-tuple of all_data, raw_data, and classifications where all_data is the assembly of raw_data and
        classifications.
        """
        if feature_type.lower() == 'rwe':
            raw_data, classification_data = Utils.load_rwe_features(datadir, kwargs.get('windowsize', 256), kwargs.get('datapoints', 1024))
        if feature_type.lower() == 'gist':
            raw_data, classification_data = Utils.load_gist_features(datadir)
        if filterhashes:
            hashes = pd.read_csv(os.path.join(datadir, 'hashes.txt'), header=None).values[:, 0]
            raw_data = Utils.filter_hashes(raw_data, hashes)
            classification_data = Utils.filter_hashes(classification_data, hashes)
        return Utils.sanity_check_classifications(raw_data, classification_data)

    @staticmethod
    def load_rwe_features(datadir, windowsize=256, datapoints=512):
        """
        Loads the data saved from preprocessing.

        :param datadir:  The data directory that contains the data.
        :param windowsize:  The window size for the RWE to load.
        :param datapoints:  The number of datapoints to load.
        :return:  raw_data, classifications tuple
        """
        # Check to see that the data directory exists, this will throw an
        # exception if it does not exist.
        os.stat(datadir)
        df = pd.read_hdf(os.path.join(datadir, 'rwe_window_{0}_datapoints_{1}.hdf'.format(windowsize, datapoints)), 'rwe')
        classifications = pd.read_hdf(os.path.join(datadir, 'classifications.hdf'), 'data')
        return df, classifications

    @staticmethod
    def load_gist_features(datadir):
        """
        Loads the data saved from preprocessing.

        :param datadir:  The data directory that contains the data.
        :return:  raw_data, classifications tuple
        """
        # Check to see that the data directory exists, this will throw an
        # exception if it does not exist.
        os.stat(datadir)
        df = pd.read_hdf(os.path.join(datadir, 'gist.hdf'), 'gist')
        classifications = pd.read_hdf(os.path.join(datadir, 'classifications.hdf'), 'data')
        return df, classifications

    @staticmethod
    def sanity_check_classifications(raw_data, classifications):
        """
        Verify that all raw data has a classification and all classifications
        have a raw data point.

        :param raw_data:  A DataFrame containing the raw data, with the hash
        as the index.  The hash must be upper case.
        :param classifications:  A dataframe containing the classifications,
        with the hash as the index.
        :return:  all_data, raw_data, classification.  all_data will have
        the classifications and raw data in the DataFrame, raw_data will only
        have the raw data, and classifications will have the classifications.
        all_data is created from raw_data and classifications.
        """
        classifications = classifications[~classifications.index.duplicated(keep='first')]
        raw_data = raw_data[~raw_data.index.duplicated(keep='first')]
        classifications = classifications.loc[classifications.index.isin(raw_data.index)]
        raw_data = raw_data.loc[raw_data.index.isin(classifications.index)]
        all_data = pd.concat([raw_data, classifications], axis=1, sort=False)
        return all_data, raw_data, classifications

    @staticmethod
    def filter_hashes(data, hashes):
        """
        Filters the data so that only hashes in the hashes list are included.

        :param data: A DataFrame where the index is hashes.
        :param hashes: A list of hashes to include for the filter.
        :return: A DataFrame of the filtered data.
        """
        return data.loc[data.index.isin(hashes)]