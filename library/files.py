# Files module
import os
import magic
import re
import pefile
import hashlib
import pickle
import gzip
import array
import io
from . import entropy
from hashlib import sha256
import pandas as pd
import numpy as np
from PIL import Image
import leargist
import tempfile
import scipy.misc


class Sample(object):
    def __init__(self, fromfile=None, frommemory=None, *args, **kwargs):
        """
        An object representing a sample for analysis.

        :param fromfile:  The file name to read.
        :param frommemory:  The data in memory for the sample.
        :param args:  Not currently used.
        :param kwargs:   Not currently used.
        """
        self.rawdata = None
        self.filename = None
        if fromfile:
            self.fromfile(fromfile)
        elif frommemory:
            self.frommemory(frommemory)
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
        """
        Calculates the running window entropy of the sample.

        :param window_size:  The window size of the RWE.
        :param normalize:  Normalize the RWE between 0 and 8.
        :return: The running window entropy as a Series.
        """
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
        return sha256_memory(self.rawdata)

    @property
    def image_data(self):
        """
        Algorithm discussed on http://sarvamblog.blogspot.com/2014/08/supervised-classification-with-k-fold.html

        :return:  The GIST image data as a Pandas Series.
        """
        ln = len(self.rawdata)
        width = 256
        rem = ln % width
        a = array.array("B")
        a.fromstring(self.rawdata[0:ln-rem])
        g = np.reshape(a, (int(len(a) / width), width))
        g = np.uint8(g)
        filename = '{0}.png'.format(self.sha256)
        scipy.misc.imsave(filename, g)

        im = Image.open(filename)
        im1 = im.resize((64, 64), Image.ANTIALIAS);
        im.close()
        des = leargist.color_gist(im1)
        X = pd.Series(des[0:320])
        X.name = self.sha256
        os.remove(filename)
        return X


def sha256_file(filename):
    """
    Calculates the SHA256 of a file.

    :param filename:  The file to calculate.
    :return:  The SHA256
    """
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            buf = f.read()
            return sha256_memory(buf)
    else:
        return ""


def sha256_memory(data):
    """
    Calculates the SHA256 from data.

    :param data:  The data to calculate.
    :return:  The SHA256
    """
    hasher = sha256()
    hasher.update(data)
    return hasher.hexdigest().upper()
