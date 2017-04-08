# Files module
import os
import magic
from . import entropy


class FileObject(object):
    def __init__(self, filename):
        """
        Creates a file object for a malware sample.

        :param filename:  The file name of the available malware sample.
        """
        if not os.path.exists(filename):
            raise ValueError("File {0} does not exist!".format(filename))

        # Default settings of members
        self.running_entropy_data = None
        self.running_entropy_window_size = 0
        self.file_size = 0

        # Fill out other data here...
        self.filename = filename
        self.data = list()
        self.filetype = magic.from_file(self.filename)
        self._read_file()
        self._parse_file_type()

    def _read_file(self):
        """
        Reads a file into a list of bytes.

        :return:  Nothing.
        """
        with open(self.filename, 'rb') as f:
            byte = f.read(1)
            while byte != b"":
                self.data.append(byte)
                byte = f.read(1)
        self.file_size = len(self.data)

    def _parse_file_type(self):
        """
        Parses this file into its appropriate type.
        
        :return:  Nothing. 
        """
        pass

    def running_entropy(self, window_size=256, normalize=True):
        """
        Calculates the running entropy of the whole file object using a
        window size.

        :param window_size:  The running window size in bytes.
        :param normalize:  True if the output should be normalized
            between 0 and 1.
        :return: A list of running entropy values for the given window size.
        """
        re = entropy.RunningEntropy(window=window_size, normalize=normalize)
        self.running_entropy_window_size = window_size
        self.running_entropy_data = re.calculate(self.data)
        return self.running_entropy_data

    def entropy(self, normalize=True):
        """
        Calculates the entropy for the whole file.

        :param normalize:  True if the output should be normalized
            between 0 and 1.
        :return: An entropy value of the whole file.
        """
        re = entropy.RunningEntropy(window=len(self.data), normalize=normalize)
        self.entropy = re.calculate(self.data)[0]
        return self.entropy
