import argparse
import os
from hashlib import sha256
from virus_total_apis import IntelApi


def main():
    parser = argparse.ArgumentParser(
        description='Downloads samples from VT Intelligence based upon a query.')
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
                        type=int, default=10, required=True)

    args = parser.parse_args()

    try:
        os.stat(args.OutputDirectory)
    except:
        os.mkdir(args.OutputDirectory)

    api = IntelApi(args.apikey)

    downloads = 0
    nextpage = None

    while downloads <= args.number_of_samples:
        nextpage, results = api.get_hashes_from_search(args.Query, nextpage)
        results = results.json()
        print("Downloading Samples...")

        for hash in results['hashes']:
            filename = os.path.join(args.OutputDirectory,
                                    hash.upper())
            try:
                os.stat(args.OutputDirectory)
            except:
                os.mkdir(args.OutputDirectory)

            print("Downloading {0}".format(hash))
            response = api.get_file(hash, args.OutputDirectory)
            print("\tDownloaded {0}".format(hash))
            print("\tVerifying hash...")
            expected_hash = hash.upper()
            dl_hash = sha256_file(filename).upper()

            if expected_hash != dl_hash:
                print("**** DOWNLOAD ERROR!  SHA256 Does not match!")
                print("\tExpected SHA256: {0}".format(expected_hash))
                print("\tCalculated SHA256: {0}".format(dl_hash))
                print("\tHave you exceeded your quota?")
            else:
                print("\t\tHash verified!")
                downloads += 1
                if downloads >= args.number_of_samples:
                    break

        if nextpage is None or downloads >= args.number_of_samples:
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
