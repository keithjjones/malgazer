import argparse
from library.files import FileObject
import numpy
import os
import time
import shutil


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Calculates the entropy of a file.')
    parser.add_argument('MalwareDirectory',
                        help='The directory containing malware to analyze.')
    parser.add_argument('DataDirectory',
                        help='The directory that will contain the data files.')
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "Multiple windows can be identified as comma "
                             "separated values without spaces."
                             "", type=str, required=False)
    parser.add_argument("-n", "--nonormalize", action='store_true',
                        help="Disables entropy normalization."
                             "", required=False)
    parser.add_argument("-m", "--maxsamples", type=int, default=0,
                        help="Maximum number of samples to process, zero for all."
                             "", required=False)
    args = parser.parse_args()

    # Normalize setup...
    if args.nonormalize:
        normalize = False
    else:
        normalize = True

    # Find window sizes
    windows = None
    if args.window:
        windows = args.window.split(',')
        windows = [x.strip() for x in windows]
        windows = [int(x) for x in windows]

    # Delete old data...
    try:
        shutil.rmtree(args.DataDirectory, ignore_errors=True)
    except:
        pass

    # Create DB directory...
    try:
        os.mkdir(args.DataDirectory)
    except:
        pass

    # # Create the DB directory if needed...
    # try:
    #     os.stat(args.DBDirectory)
    # except:
    #     os.mkdir(args.DBDirectory)

    # Crawl the directories for malware
    samples_processed = 0
    for root, dirs, files in os.walk(args.MalwareDirectory):
        for f in files:
            subdir = root[len(args.MalwareDirectory):]

            # Create the malware file name...
            malwarepath = os.path.join(root, f)
            try:
                m = FileObject(malwarepath)
            except:
                continue

            print("\tCalculating: {0} Type: {1}".format(m.filename, m.filetype))

            # Create the DB file name...
            datadir = os.path.join(args.DataDirectory, subdir)
            picklefile = os.path.join(datadir, f) + ".pickle.gz"

            print("\tSaving data to {0}".format(picklefile))

            # Create the directory if needed...
            try:
                os.stat(datadir)
            except:
                os.mkdir(datadir)

            # Calculate the entropy of the file...
            fileentropy = m.entropy(normalize)

            # Calculate the window entropy for malware samples...
            if windows is not None:
                # Iterate through the window sizes...
                for w in windows:
                    if w < m.file_size:
                        print("\t\tCalculating window size {0:,}".format(w))

                        # Calculate running entropy...
                        rwe = m.running_entropy(w, normalize)

                # Write the running entropy...
                m.write(picklefile)

            samples_processed += 1
            print("{0:n} samples processed...".format(samples_processed))
            if args.maxsamples > 0 and samples_processed >= args.maxsamples:
                break


if __name__ == "__main__":
    main()
