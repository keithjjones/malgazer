# Files module
import os
import magic
import re
import pefile
import hashlib
import pickle
import gzip
from . import entropy
from hashlib import sha256
import pandas as pd


class Sample(object):
    def __init__(self, *args, **kwargs):
        """
        An object representing a sample for analysis.

        :param args:  Not currently used.
        :param kwargs:   Not currently used.
        """
        self.rawdata = None
        self.filename = None
        # super(Sample, self).__init__(*args, **kwargs)

    def fromfile(self, filename):
        """
        Read a sample's data from a file.

        :param filename:  The file name to read.
        :return:  A binary string representing the sample's content.
        """
        with open(filename, 'rb') as myfile:
            self.rawdata = myfile.read()
        self.filename = filename
        return self.rawdata

    def frommemory(self, filestream):
        """
        Read a sample's data from a memory.

        :param filestream:  The file stream to store.
        :return:  A binary string representing the sample's content.
        """
        self.rawdata = filestream
        return self.rawdata

    def running_window_entropy(self, window_size=256, normalize=True):
        e = entropy.RunningEntropy()
        rwe = e.calculate(self.data, window_size, normalize)
        ds = pd.Series(rwe)
        ds.name = self.sha256
        return ds

    @property
    def data(self):
        """
        Returns a bytearray of the sample's data.

        :return:  A bytearray of the sample's data.
        """
        return bytearray(self.rawdata)

    @property
    def sha256(self):
        """
        Calculates the SHA256 of a file by filename.

        :return:  The SHA256 if successful, None otherwise.
        """
        return sha256(self.rawdata)


def sha256_file(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            buf = f.read()
            return sha256(buf)
    else:
        return ""


def sha256(data):
    """
    Calculates the SHA256 from data.

    :param data:  The data to calculate.
    :return:  The SHA256
    """
    hasher = sha256()
    hasher.update(data)
    return hasher.hexdigest().upper()
