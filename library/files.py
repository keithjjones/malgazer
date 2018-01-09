# Files module
import os
import magic
import re
import pefile
import hashlib
from . import entropy


class FileObject(object):

    # Below, the format is matching RE, file type,
    # and optional function to run on the file name
    FILE_TYPES = [
        ["PE.*MS Windows.*", "pefile", pefile.PE]
    ]

    def __init__(self, filename):
        """
        Creates a file object for a malware sample.

        :param filename:  The file name of the available malware sample.
        """
        if not os.path.exists(filename):
            raise ValueError("File {0} does not exist!".format(filename))

        # Default settings of members
        self.runningentropy = entropy.RunningEntropy()
        self.file_size = 0
        # The parsed info for the file...
        self.parsedfile = None

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
            hash_md5 = hashlib.md5()
            hash_sha256 = hashlib.sha256()
            byte = f.read(1)
            while byte != b"":
                hash_md5.update(byte)
                hash_sha256.update(byte)
                self.data.append(byte)
                byte = f.read(1)
        self.file_size = len(self.data)
        self.md5 = hash_md5.hexdigest()
        self.sha256 = hash_sha256.hexdigest()

    def _parse_file_type(self):
        """
        Parses this file into its appropriate type.
        
        :return:  Nothing. 
        """
        for FILE_TYPE in self.FILE_TYPES:
            if re.match(FILE_TYPE[0], self.filetype):
                self.parsedfile = {"type": FILE_TYPE[1]}
                if (len(FILE_TYPE) > 2 and FILE_TYPE[2] is not None
                        and callable(FILE_TYPE[2])):
                    self.parsedfile['file'] = FILE_TYPE[2](self.filename)

    # TODO: Fix this up later, if I decide I need it again...
    # def parsed_file_running_entropy(self, window_size=256, normalize=True):
    #     """
    #     Calculates the running entropy of the file with respect to the file
    #     type.  For example, Windows PE files entropy will be
    #     calculated on each section.
    #
    #     :param window_size:  The running window size in bytes.
    #     :param normalize:   True if the output should be normalized between
    #         0 and 1.
    #     :return: A dict with running window entropy and other metadata
    #         as appropriate for the file type.
    #         Returns None if the file was not parsed successfully.
    #     """
    #     # Return right away if the data was not parsed...
    #     if self.parsedfile is None:
    #         return None
    #
    #     # Windows PE Files...
    #     if self.parsedfile['type'] == 'pefile':
    #         self.parsedfile['running_entropy'] = dict()
    #         self.parsedfile['running_entropy']['sections'] = list()
    #         for section in self.parsedfile['file'].sections:
    #             section_name = section.Name.decode('UTF-8').rstrip('\x00')
    #             offset = section.PointerToRawData
    #             length = section.SizeOfRawData
    #             # TODO: Above may be section.Misc_VirtualSize - test this.
    #             # More info:
    #             # https://msdn.microsoft.com/en-us/library/ms809762.aspx
    #             self.parsedfile['running_entropy']['sections'].append({
    #                 'name': section_name, 'offset': offset, 'length': length,
    #                 'entropy_window_length': window_size,
    #                 'normalize': normalize,
    #                 'running_entropy': self.running_entropy(window_size,
    #                                                         normalize,
    #                                                         offset=offset,
    #                                                         length=length)})
    #         return self.parsedfile['running_entropy']

    def running_entropy(self, window_size=256, normalize=True):
        """
        Calculates the running entropy of the whole generic file object using a
        window size.

        :param window_size:  The running window size in bytes.
        :param normalize:  True if the output should be normalized
            between 0 and 1.
        :return: A list of running entropy values for the given window size.
        """
        entropy = self.runningentropy.calculate(self.data,
                                                window=window_size,
                                                normalize=normalize)
        if len(entropy) != self.file_size - window_size + 1:
            raise Exception("This should not happen.  Check this code.")
        return entropy

    def entropy(self, normalize=True):
        """
        Calculates the entropy for the whole file.

        :param normalize:  True if the output should be normalized
            between 0 and 1.
        :return: An entropy value of the whole file.
        """
        entropy = self.runningentropy.calculate(self.data,
                                                window=len(self.data),
                                                normalize=normalize)
        if len(entropy) > 1:
            raise Exception("This should not happen.  Check this code.")

        return entropy[0]

    def write_entropy(self, sqlite_filename):
        self.runningentropy.write(sqlite_filename)

    def read_entropy(self, sqlite_filename):
        self.runningentropy.read(sqlite_filename)
