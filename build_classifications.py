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
output_csv = '/Dirty/Samples/vt_focused_classifications.csv'
outputs = []

for s in subdirs:
  outputs.append(pd.read_csv(os.path.join(s, 'classifications_trimmed.csv'), index_col=0))    
  
output_df = pd.concat(outputs)
output_df = output_df[~output_df.index.duplicated(keep='first')]
output_df['classification'].value_counts()
output_df.to_csv(output_csv)