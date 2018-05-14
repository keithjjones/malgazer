# Install the plaidml backend
#import plaidml.keras
#plaidml.keras.install_backend()

import batch_preprocess_entropy
from library.utils import Utils
from library.ml import ML
import pandas as pd
import numpy as np
import time
import os

pd.set_option('max_colwidth', 64)

#
# This will process a specific directories into master files in each directory
#

subdirs = [
        '/Dirty/Samples/VT_20180514.1/'       
        ]

for s in subdirs:
    classifications = Utils.estimate_vt_classifications_from_csv(os.path.join(s, 'all_vt_data.csv'))
    classifications.to_csv(os.path.join(s, 'classifications.csv'))
    classes = ['Worm', 'Trojan', 'Backdoor', 'Virus', 'PUA', 'Ransom']
    c = classifications[classifications['classification'].isin(classes)]
    c = c[~c.index.duplicated(keep='first')]
    c.to_csv(os.path.join(s, 'classifications_trimmed.csv'))
    c['classification'].value_counts()

#
# This merges all data together from separate directories.
#

subdirs = [
        '/Dirty/Samples/VT_20180503.1/', 
        '/Dirty/Samples/VT_20180504.1/', 
        '/Dirty/Samples/VT_20180505.1/', 
        '/Dirty/Samples/VT_20180506.1/', 
        '/Dirty/Samples/VT_20180507.1/',
        '/Dirty/Samples/VT_20180508.1/',
        '/Dirty/Samples/VT_20180508.2/',
        '/Dirty/Samples/VT_20180509.1/',
        '/Dirty/Samples/VT_20180510.1/',
        '/Dirty/Samples/VT_20180512.1/',
        '/Dirty/Samples/VT_20180514.1/'
        ]
output_dir = '/Dirty/Samples'
output_csv = os.path.join(output_dir, 'vt_focused_classifications.csv')
output_csv_trimmed = os.path.join(output_dir, 'vt_focused_classifications_trimmed.csv')
outputs = []

for s in subdirs:
  outputs.append(pd.read_csv(os.path.join(s, 'classifications_trimmed.csv'), index_col=0))    
  
output_df = pd.concat(outputs)
output_df = output_df[~output_df.index.duplicated(keep='first')]
output_df['classification'].value_counts()
output_df.to_csv(output_csv)
output_df_trimmed = output_df.groupby(['classification']).head(11000)
output_df_trimmed.to_csv(output_csv_trimmed)

hashes_csv = os.path.join(output_dir, 'vt_focused_hashes_trimmed.csv')
hashes = pd.DataFrame(output_df_trimmed.index.values)
hashes.to_csv(hashes_csv)

worm = output_df_trimmed[output_df_trimmed['classification'] == 'Worm']
worm.to_csv(os.path.join(output_dir, 'worm.csv'))
worm_hashes = pd.DataFrame(worm.index.values)
worm_hashes.to_csv(os.path.join(output_dir, 'worm_hashes.csv'))

trojan = output_df_trimmed[output_df_trimmed['classification'] == 'Trojan']
trojan.to_csv(os.path.join(output_dir, 'trojan.csv'))
trojan_hashes = pd.DataFrame(trojan.index.values)
trojan_hashes.to_csv(os.path.join(output_dir, 'trojan_hashes.csv'))

backdoor = output_df_trimmed[output_df_trimmed['classification'] == 'Backdoor']
backdoor.to_csv(os.path.join(output_dir, 'backdoor.csv'))
backdoor_hashes = pd.DataFrame(backdoor.index.values)
backdoor_hashes.to_csv(os.path.join(output_dir, 'backdoor_hashes.csv'))

virus = output_df_trimmed[output_df_trimmed['classification'] == 'Virus']
virus.to_csv(os.path.join(output_dir, 'virus.csv'))
virus_hashes = pd.DataFrame(virus.index.values)
virus_hashes.to_csv(os.path.join(output_dir, 'virus_hashes.csv'))

pua = output_df_trimmed[output_df_trimmed['classification'] == 'PUA']
pua.to_csv(os.path.join(output_dir, 'pua.csv'))
pua_hashes = pd.DataFrame(pua.index.values)
pua_hashes.to_csv(os.path.join(output_dir, 'pua_hashes.csv'))

ransom = output_df_trimmed[output_df_trimmed['classification'] == 'Ransom']
ransom.to_csv(os.path.join(output_dir, 'ransom.csv'))
ransom_hashes = pd.DataFrame(ransom.index.values)
ransom_hashes.to_csv(os.path.join(output_dir, 'ransom_hashes.csv'))
