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
        nextpage, results = api.get_intel_notifications_feed(nextpage)
        results = results.json()
        print("Downloading {0} Samples..."
              .format(len(results['notifications'])))

        for notification in results['notifications']:
            downloads += 1
            subdir = os.path.join(args.OutputDirectory,
                                  notification['ruleset_name'])
            try:
                os.stat(subdir)
            except:
                os.mkdir(subdir)

            try:
                print(sha256_file(os.path.join(subdir, notification['sha256'])).upper())
            except:
                pass

            print("Downloading {0}".format(notification['sha256']))

            if (os.path.isfile(os.path.join(subdir, notification['sha256']))
                and notification['sha256'].upper() ==
                    sha256_file(
                        os.path.join(subdir, notification['sha256'])).upper()):
                print("\tFile {0} Already Downlaoded!"
                      .format(notification['sha256']))
            else:
                api.get_file(notification['sha256'], subdir)
                print("\tDownloaded {0}".format(notification['sha256']))

        if nextpage is None:
            break

    print("Downloaded {0} Total Samples".format(downloads))


def sha256_file(filename):
    hasher = sha256()
    with open(filename,'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

if __name__ == "__main__":
    main()
