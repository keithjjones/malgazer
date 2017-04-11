import argparse
import os
from hashlib import sha256
from virus_total_apis import IntelApi


def main():
    parser = argparse.ArgumentParser(
        description='Downloads samples from VT Intelligence.')
    parser.add_argument('OutputDirectory',
                        help='The output directory for the samples.')
    parser.add_argument("-a", "--apikey",
                        help="Your VT Intelligence API key."
                             "", required=True)

    args = parser.parse_args()

    try:
        os.stat(args.OutputDirectory)
    except:
        os.mkdir(args.OutputDirectory)

    api = IntelApi(args.apikey)

    downloads = 0
    nextpage = None

    while True:
        samplestodelete = []
        nextpage, results = api.get_intel_notifications_feed(nextpage)
        results = results.json()
        print("Downloading {0} Samples..."
              .format(len(results['notifications'])))

        for notification in results['notifications']:
            downloads += 1
            subdir = os.path.join(args.OutputDirectory,
                                  notification['ruleset_name'])
            filename = os.path.join(subdir, notification['sha256'])

            try:
                os.stat(subdir)
            except:
                os.mkdir(subdir)

            print("Downloading {0}".format(notification['sha256']))
            response = api.get_file(notification['sha256'], subdir)
            print("\tDownloaded {0}".format(notification['sha256']))
            expected_hash = notification['sha256'].upper()
            dl_hash = sha256_file(filename).upper()

            if expected_hash != dl_hash:
                print("**** DOWNLOAD ERROR!  SHA256 Does not match!")
                print("\tExpected SHA256: {0}".format(expected_hash))
                print("\tCalculated SHA256: {0}".format(dl_hash))
                print("\tWill not delete this sample from the feed.")
                print("\tHave you exceeded your quota?")
            else:
                samplestodelete.append(notification['id'])

        if len(samplestodelete) > 0:
            api.delete_intel_notifications(samplestodelete)
            print("Deleted {0} Samples From Feed".format(len(samplestodelete)))

        if nextpage is None:
            break

    print("Downloaded {0} Total Samples".format(downloads))


def sha256_file(filename):
    hasher = sha256()
    with open(filename,'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

if __name__ == "__main__":
    main()
