import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Pulls hashes from VT data.')
    parser.add_argument('VTCSV',
                        help='The VT CSV file.')
    parser.add_argument("-l", "--lowercase",
                        help="Use lower case for hashes."
                             "", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.VTCSV, index_col=0)

    for index, row in df.iterrows():
        if args.lowercase:
            print(index.lower())
        else:
            print(index.upper())


if __name__ == "__main__":
    main()
