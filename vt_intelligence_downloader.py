import argparse
import datetime
import time
import os
from virus_total_apis import IntelApi, PublicApi, PrivateApi
from library.files import sha256_file
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Downloads samples from VT Intelligence based upon a query.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('OutputDirectory',
                        help='The output directory for the samples.')
    parser.add_argument('Query',
                        help='The valid VTI query.')
    parser.add_argument("-a", "--apikey",
                        help="Your VT Intelligence API key."
                             "", required=True)
    parser.add_argument("-n", "--number_of_samples",
                        help="The number of files to download."
                             "",
                        type=int, default=50, required=True)

    args = parser.parse_args()

    try:
        os.stat(args.OutputDirectory)
    except:
        os.makedirs(args.OutputDirectory)

    intel_api = IntelApi(args.apikey)
    public_api = PublicApi(args.apikey)
    # private_api = PrivateApi(args.apikey)

    downloads = 0
    nextpage = None

    df = pd.DataFrame()
    rows_to_add = []

    while downloads <= args.number_of_samples:
        try:
            results = None
            while results is None:
                nextpage, results = intel_api.get_hashes_from_search(args.Query, nextpage)
                if results.status_code != 200:
                    print("\tError downloading hashes, retrying...")
                    time.sleep(60)
                    results = None
                else:
                    results = results.json()

            print("Downloading hashes for samples...")

            for hash in results['hashes']:
                if downloads < args.number_of_samples:
                    filename = os.path.join(args.OutputDirectory,
                                            hash.upper())
                    try:
                        os.stat(args.OutputDirectory)
                    except:
                        os.makedirs(args.OutputDirectory)

                    if not os.path.isfile(filename):
                        print("Downloading {0}".format(hash))
                        downloaded = False
                        while downloaded is False:
                            try:
                                response = intel_api.get_file(hash, args.OutputDirectory)
                            except KeyboardInterrupt:
                                if os.path.isfile(filename):
                                    os.remove(filename)
                                raise

                            print("\t\tDownloaded {0}".format(hash))
                            print("\t\tVerifying hash...")
                            expected_hash = hash.upper()
                            dl_hash = sha256_file(filename).upper()

                            if expected_hash != dl_hash:
                                print("\t**** DOWNLOAD ERROR!  SHA256 Does not match!")
                                print("\t\tExpected SHA256: {0}".format(expected_hash))
                                print("\t\tCalculated SHA256: {0}".format(dl_hash))
                                print("\t\tHave you exceeded your quota?")
                            else:
                                print("\t\t\tHash verified!")
                                downloads += 1
                                print("\t\tDownloaded {0:,} samples...".format(downloads))
                                downloaded = True

                        file_report = None
                        while file_report is None:
                            print("\t\tDownloading file report...")
                            file_report = public_api.get_file_report(hash)
                            if 'error' in file_report:
                                print("\t\t\t\tError, retrying...")
                                time.sleep(60)
                                file_report = None

                    ds = pd.Series(file_report['results'])
                    ds.name = hash.upper()
                    rows_to_add.append(ds)
                else:
                    break

            if nextpage is None or downloads >= args.number_of_samples:
                break
        except KeyboardInterrupt:
            print("Caught CTRL-C!")
            break

    print("Assembling CSV...")
    df = df.append(rows_to_add)

    now = datetime.datetime.now()
    now_str = "{0}_{1:02}_{2:02}_{3:02}_{4:02}_{5:02}_{6}".format(now.year,
                                                                  now.month,
                                                                  now.day,
                                                                  now.hour,
                                                                  now.minute,
                                                                  now.second,
                                                                  now.microsecond)
    print("Writing metadata CSV...")
    df.to_csv(os.path.join(args.OutputDirectory, "vti_metadata_{0}.csv".format(now_str)))
    print("Downloaded {0:,} Total Samples".format(downloads))


if __name__ == "__main__":
    main()
