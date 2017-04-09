# Files module
import os
import magic
import re
import pefile
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
        # Detect Windows PE Files...
        if re.match("PE.*MS Windows.*", self.filetype):
            self.parsedfile = {"type": "pefile",
                               "file": pefile.PE(self.filename)}

    def parsed_file_running_entropy(self, window_size=256, normalize=True):
        """
        Calculates the running entropy of the file with respect to the file
        type.  For example, Windows PE files entropy will be 
        calculated on each section.
        
        :param window_size:  The running window size in bytes. 
        :param normalize:   True if the output sould be normalized between
            0 and 1.
        :return: A dict of running window entropy lists as appropriate for
            the file type.  None if the file was not parsed successfully.
        """
        if self.parsedfile is None:
            return None

        # Windows PE Files...
        if self.parsedfile['type'] == 'pefile':
            self.parsedfile['running_entropy_sections'] = dict()
            for section in self.parsedfile['file'].sections:
                section_name = section.Name.decode('UTF-8').rstrip('\x00')
                offset = section.PointerToRawData
                length = section.Misc_VirtualSize
                # TODO: Above may be section.SizeOfRawData - test this.
                self.parsedfile['running_entropy_sections'][section_name] = \
                    self.running_entropy(window_size, normalize,
                                         offset=offset, length=length)
            return self.parsedfile['running_entropy_sections']

    def running_entropy(self, window_size=256, normalize=True,
                        offset=0, length=None):
        """
        Calculates the running entropy of the whole generic file object using a
        window size.  Optionally, you can specify an offset and length
        within the file for the calculations.

        :param window_size:  The running window size in bytes.
        :param normalize:  True if the output should be normalized
            between 0 and 1.
        :param offset:  Byte offset to start calculations within the file.
            Note, use a valid value or there can be an exception!
        :param length:  Length, in bytes, of the data area to compute.
            Set to None to ignore the length.  Note, use a valid value or
            there can be an exception!
        :return: A list of running entropy values for the given window size.
        """
        runent = entropy.RunningEntropy(window=window_size, normalize=normalize)

        if offset > self.file_size - window_size or offset < 0:
            raise IndexError("The offset {0} is not a valid "
                             "value for file length {1} "
                             "and window size {2}!"
                             .format(offset, self.file_size, window_size))

        if length is not None:
            if length + offset > self.file_size:
                raise IndexError("The length {0} is not a valid "
                                 "value for file length {1} "
                                 "and offset {2}!"
                                 .format(length, self.file_size, offset))
        else:
            length = self.file_size - offset

        data = self.data[offset:length+offset]
        self.running_entropy_offset = offset
        self.running_entropy_length = length

        self.running_entropy_window_size = window_size
        self.running_entropy_data = runent.calculate(data)
        return self.running_entropy_data

    def entropy(self, normalize=True):
        """
        Calculates the entropy for the whole file.

        :param normalize:  True if the output should be normalized
            between 0 and 1.
        :return: An entropy value of the whole file.
        """
        runent = entropy.RunningEntropy(window=len(self.data), normalize=normalize)
        self.entropy = runent.calculate(self.data)[0]
        return self.entropy
