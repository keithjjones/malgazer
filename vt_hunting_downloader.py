import argparse
import datetime
import os
import time
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
                        help="Delete downloaded samples from feed."
                             "", action='store_true')
    parser.add_argument("-dn", "--delete_non_matches",
                        help="Delete samples that do not match from feed."
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
        samplestodelete = []
        results = None
        while results is None:
            results = api.get_intel_notifications_feed(nextpage)
            nextpage = results['results']['next']
            results = results['results']
            if 'error' in results:
                print("\t\t\tError, retrying...")
                time.sleep(60)
                results = None

        print("Downloading hashes for samples...")

        for notification in results['notifications']:
            if notification['positives'] >= args.positives:
                subdir = os.path.join(args.OutputDirectory,
                                      notification['ruleset_name'])
                filename = os.path.join(subdir, notification['sha256'])

                try:
                    os.stat(subdir)
                except:
                    os.mkdir(subdir)

                print("Downloading {0}".format(notification['sha256']))
                downloaded = False
                while downloaded is False:
                    response = api.get_file(notification['sha256'], subdir)
                    print("\tDownloaded {0}".format(notification['sha256']))
                    print("\tVerifying hash...")
                    expected_hash = notification['sha256'].upper()
                    dl_hash = sha256_file(filename).upper()

                    if expected_hash != dl_hash:
                        print("**** DOWNLOAD ERROR!  SHA256 Does not match!")
                        print("\tExpected SHA256: {0}".format(expected_hash))
                        print("\tCalculated SHA256: {0}".format(dl_hash))
                        print("\tWill not delete this sample from the feed.")
                        print("\tHave you exceeded your quota?")
                    else:
                        print("\t\tHash verified!")
                        downloaded = True
                        samplestodelete.append(notification['id'])

                downloads += 1

                print("\tDownloaded {0} samples...".format(downloads))

                ds = pd.Series(notification)
                ds.name = notification['sha256']
                df = df.append(ds)
            else:
                if args.delete_non_matches:
                    # Delete the notification if it does not match
                    samplestodelete.append(notification['id'])

            if args.number_of_samples > 0 and downloads >= args.number_of_samples:
                break

        if nextpage is None or (args.number_of_samples > 0 and
                                downloads >= args.number_of_samples):
            break

    if len(samplestodelete) > 0 and args.delete_downloaded:
        api.delete_intel_notifications(samplestodelete)
        print("Deleted {0} Samples From Feed".format(len(samplestodelete)))

    now = datetime.datetime.now()
    now_str = "{0}_{1:02}_{2:02}_{3:02}_{4:02}_{5:02}_{6}".format(now.year,
                                                                  now.month,
                                                                  now.day,
                                                                  now.hour,
                                                                  now.minute,
                                                                  now.second,
                                                                  now.microsecond)
    df.to_csv(os.path.join(args.OutputDirectory, "vti_metadata_{0}.csv".format(now_str)))
    print("Downloaded {0} Total Samples".format(downloads))


if __name__ == "__main__":
    main()
