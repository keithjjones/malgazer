import argparse
import datetime
import os
import time
import ast
import json
import pandas as pd
from library.files import sha256_file
from virus_total_apis import IntelApi


def main():
    parser = argparse.ArgumentParser(
        description='Downloads samples from VT Intelligence.')
    parser.add_argument('OutputDirectory',
                        help='The output directory for the samples.')
    parser.add_argument("-a", "--apikey",
                        help="Your VT Intelligence API key."
                             "", required=True)
    parser.add_argument("-p", "--positives",
                        help="Detections must have at least this many positives."
                             "",
                        type=int, default=25)
    parser.add_argument("-n", "--number_of_samples",
                        help="The number of files to download.  "
                             "Set to zero for all downloads.",
                        type=int, default=0)
    parser.add_argument("-d", "--delete_downloaded",
                        help="Delete downloaded samples and metadata from feed."
                             "", action='store_true')
    parser.add_argument("-dn", "--delete_non_matches",
                        help="Delete samples that do not match from feed."
                             "", action='store_true')
    parser.add_argument("-dd", "--dont_download_sample",
                        help="Enable to just get metadata, without downloading samples."
                             "", action='store_true')
    args = parser.parse_args()

    try:
        os.stat(args.OutputDirectory)
    except:
        os.makedirs(args.OutputDirectory)

    api = IntelApi(args.apikey)

    downloads = 0
    nextpage = None

    df = pd.DataFrame()

    while True:
        try:
            results = None
            while results is None:
                results = api.get_intel_notifications_feed(nextpage)
                nextpage = results['results']['next']
                results = results['results']
                if 'error' in results:
                    print("\tError downloading hashes, retrying...")
                    time.sleep(60)
                    results = None

            print("Downloading hashes for samples...")

            for notification in results['notifications']:
                if int(notification['positives']) >= args.positives:
                    subdir = os.path.join(args.OutputDirectory,
                                          notification['ruleset_name'],
                                          notification['subject'])
                    filename = os.path.join(subdir, notification['sha256'])

                    if not os.path.isfile(filename):
                        # Make the directory
                        try:
                            os.stat(subdir)
                        except:
                            os.makedirs(subdir)
                        print("\tDownloading {0}".format(notification['sha256']))
                        if not args.dont_download_sample:
                            downloaded = False
                            while downloaded is False:
                                try:
                                    response = api.get_file(notification['sha256'], subdir)
                                except KeyboardInterrupt:
                                    if os.path.isfile(filename):
                                        os.remove(filename)
                                    raise
                                print("\t\tDownloaded {0}".format(notification['sha256']))
                                print("\t\tVerifying hash...")
                                expected_hash = notification['sha256'].upper()
                                dl_hash = sha256_file(filename).upper()

                                if expected_hash != dl_hash:
                                    print("\t**** DOWNLOAD ERROR!  SHA256 Does not match!")
                                    print("\t\tExpected SHA256: {0}".format(expected_hash))
                                    print("\t\tCalculated SHA256: {0}".format(dl_hash))
                                    print("\t\tWill not delete this sample from the feed.")
                                    print("\t\tHave you exceeded your quota?")
                                else:
                                    print("\t\t\tHash verified!")
                                    downloaded = True
                                    if args.delete_downloaded:
                                        print("\t\tDeleting downloaded sample from feed...")
                                        del_response = api.delete_intel_notifications([notification['id']])
                        else:
                            print("\t\tSkipping sample download, downloading metadata...")
                            if args.delete_downloaded:
                                print("\t\tDeleting downloaded sample from feed...")
                                del_response = api.delete_intel_notifications([notification['id']])

                        downloads += 1
                        print("\t\tDownloaded {0:,} samples...".format(downloads))

                    else:
                        print("\tDeleting duplicate sample from feed...")
                        if args.delete_downloaded:
                            del_response = api.delete_intel_notifications([notification['id']])

                    ds = pd.Series(notification)
                    ds.name = notification['sha256']
                    ds_scans = pd.Series(notification['scans'])
                    ds_scans.name = notification['sha256']
                    ds = ds.append(ds_scans)
                    df = df.append(ds)
                else:
                    if args.delete_non_matches:
                        # Delete the notification if it does not match
                        del_response = api.delete_intel_notifications([notification['id']])

                if args.number_of_samples > 0 and downloads >= args.number_of_samples:
                    break

            if nextpage is None or (args.number_of_samples > 0 and
                                    downloads >= args.number_of_samples):
                break
        except KeyboardInterrupt:
            print("Caught CTRL-C!")
            break

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
