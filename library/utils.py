# Utilities class
import os
import re
import pandas as pd
from .files import FileObject


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
        malware_files_re = re.compile('[a-z0-9]{64}',
                                      flags=re.IGNORECASE)
        samples_processed = 0
        for root, dirs, files in os.walk(in_directory):
            for file in files:
                if malware_files_re.match(file):
                    print("Input file: {0}\n".format(file))
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

                    samples_processed += 1
                    print("{0:n} samples processed...".format(samples_processed))