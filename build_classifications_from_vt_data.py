import argparse
import sys
from library.utils import Utils
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


pd.set_option('max_colwidth', 64)

classes = ['Worm', 'Trojan', 'Backdoor', 'Virus', 'PUA', 'Ransom']


def assemble_hdfs(hdfname):
    classifications = Utils.estimate_vt_classifications_from_hdf(hdfname)
    classifications = classifications[~classifications.index.duplicated(keep='first')]
    filename = os.path.join(os.path.dirname(hdfname), 'classifications.hdf')
    try:
        os.remove(filename)
    except:
        pass
    classifications.to_hdf(filename, 'data')
    print("\t\tWrote {0}".format(filename))
    c_trimmed = classifications[classifications['classification'].isin(classes)]
    filename = os.path.join(os.path.dirname(hdfname), 'classifications_trimmed.hdf')
    try:
        os.remove(filename)
    except:
        pass
    c_trimmed.to_hdf(filename, 'data')
    print("\t\tWrote {0}".format(filename))


def main(arguments=None):
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Assembles VT data from a directory of files created by VT downloader scripts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('DataDirectory',
                        help='The directory containing the VT data files.')
    parser.add_argument("-p", "--perclass",
                        help="The maximum number of samples, per class."
                             "", type=int, default=11000)
    parser.add_argument("-j", "--jobs",
                        help="The number of jobs for this task.  "
                        "Use -1 for all CPU cores."
                             "", type=int, default=-1)
    if isinstance(arguments, list):
        args = parser.parse_args(arguments)
    else:
        args = parser.parse_args()

    path = args.DataDirectory
    output_hdf = os.path.join(path, 'classifications_all.hdf')
    output_hdf_trimmed = os.path.join(path, 'classifications.hdf')

    try:
        os.remove(output_hdf)
    except:
        pass
    try:
        os.remove(output_hdf_trimmed)
    except:
        pass

    print("Finding VT data...")
    hdf_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('vti_metadata_') and file.endswith(".hdf"):
                hdf_files.append(os.path.join(root, file))
                print("\tFile: {0}".format(os.path.join(root, file)))

    # Old code for changing CSV files to HDF files.
    # for hdf_file in hdf_files:
    #     print("\tFile:     {0}".format(hdf_file))
    #     data = pd.read_csv(hdf_file, index_col=0)
    #     newpath = os.path.splitext(hdf_file)[0] + ".hdf"
    #     print("\tNew File: {0}".format(newpath))
    #     data.to_hdf(newpath, 'data')

    print("Assembling VT data...")
    saved_futures = {}
    max_workers = args.jobs
    if max_workers < 0:
        max_workers = None
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for hdf_file in hdf_files:
            print("\tComputing File: {0}".format(hdf_file))
            future = executor.submit(assemble_hdfs, hdf_file)
            saved_futures[future] = hdf_file
        for future in as_completed(saved_futures):
            print("\tFinished Computing File: {0}".format(saved_futures[future]))

    print("Merging VT data...")
    hdf_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'classifications_trimmed.hdf':
                hdf_files.append(os.path.join(root, file))
                print("\tFile: {0}".format(os.path.join(root, file)))

    print("Assembling master data set...")
    outputs = []
    for hdf_file in hdf_files:
        print("\tFile: {0}".format(hdf_file))
        outputs.append(pd.read_hdf(hdf_file, 'data'))

    print("Cleaning up master data set...")
    output_df = pd.concat(outputs)
    output_df = output_df[~output_df.index.duplicated(keep='first')]
    print("Full Data Set:")
    print(output_df['classification'].value_counts())

    print("Writing master data set to: {0} and {1}".format(output_hdf, output_hdf_trimmed))
    output_df.to_hdf(output_hdf, 'data')
    output_df_trimmed = output_df.groupby(['classification']).head(args.perclass)
    output_df_trimmed.to_hdf(output_hdf_trimmed, 'data')
    print("Trimmed Data Set:")
    print(output_df_trimmed['classification'].value_counts())

    # hashes_csv = os.path.join(path, 'hashes.csv')
    # hashes = pd.DataFrame(output_df_trimmed.index.values)
    # hashes.columns = ['index']
    # hashes.to_csv(hashes_csv, index=False, header=False)
    #
    # worm = output_df_trimmed[output_df_trimmed['classification'] == 'Worm']
    # worm.to_hdf(os.path.join(path, 'worm.hdf'), 'data')
    # worm_hashes = pd.DataFrame(worm.index.values)
    # worm_hashes.columns = ['sha256']
    # worm_hashes.to_hdf(os.path.join(path, 'worm_hashes.hdf'), 'data')
    #
    # trojan = output_df_trimmed[output_df_trimmed['classification'] == 'Trojan']
    # trojan.to_hdf(os.path.join(path, 'trojan.hdf'), 'data')
    # trojan_hashes = pd.DataFrame(trojan.index.values)
    # trojan_hashes.columns = ['sha256']
    # trojan_hashes.to_hdf(os.path.join(path, 'trojan_hashes.hdf'), 'data')
    #
    # backdoor = output_df_trimmed[output_df_trimmed['classification'] == 'Backdoor']
    # backdoor.to_hdf(os.path.join(path, 'backdoor.hdf'), 'data')
    # backdoor_hashes = pd.DataFrame(backdoor.index.values)
    # backdoor_hashes.columns = ['sha256']
    # backdoor_hashes.to_hdf(os.path.join(path, 'backdoor_hashes.hdf'), 'data')
    #
    # virus = output_df_trimmed[output_df_trimmed['classification'] == 'Virus']
    # virus.to_hdf(os.path.join(path, 'virus.hdf'), 'data')
    # virus_hashes = pd.DataFrame(virus.index.values)
    # virus_hashes.columns = ['sha256']
    # virus_hashes.to_hdf(os.path.join(path, 'virus_hashes.hdf'), 'data')
    #
    # pua = output_df_trimmed[output_df_trimmed['classification'] == 'PUA']
    # pua.to_hdf(os.path.join(path, 'pua.hdf'), 'data')
    # pua_hashes = pd.DataFrame(pua.index.values)
    # pua_hashes.columns = ['sha256']
    # pua_hashes.to_hdf(os.path.join(path, 'pua_hashes.hdf'), 'data')
    #
    # ransom = output_df_trimmed[output_df_trimmed['classification'] == 'Ransom']
    # ransom.to_hdf(os.path.join(path, 'ransom.hdf'), 'data')
    # ransom_hashes = pd.DataFrame(ransom.index.values)
    # ransom_hashes.columns = ['sha256']
    # ransom_hashes.to_hdf(os.path.join(path, 'ransom_hashes.hdf'), 'data')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
