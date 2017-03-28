import argparse
import os
import json
from virus_total_apis import IntelApi


def main():
    parser = argparse.ArgumentParser(
        description='Downloads samples from VT Intelligence.')
    parser.add_argument('InputJSON',
                        help='The VT Intelligence notifications JSON.')
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

    inputjson = json.loads(open(args.InputJSON).read())

    api = IntelApi(args.apikey)

    for notification in inputjson['notifications']:
        print("Downloading {0}".format(notification['sha256']))
        api.get_file(notification['sha256'], args.OutputDirectory)
        print("\tDownloaded {0}".format(notification['sha256']))

if __name__ == "__main__":
    main()