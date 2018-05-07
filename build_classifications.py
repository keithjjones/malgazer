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

subdir = '/Dirty/Samples/VT_20180505.1/'

classifications = Utils.estimate_vt_classifications_from_csv(os.path.join(subdir, 'all_vt_data.csv'))
classifications.to_csv(os.path.join(subdir, 'classifications.csv'))
classes = ['Worm', 'Trojan', 'Backdoor', 'Virus', 'PUA', 'Ransom']
c = classifications[classifications['classification'].isin(classes)]
c = c[~c.index.duplicated(keep='first')]
c.to_csv(os.path.join(subdir, 'classifications_trimmed.csv'))
c['classification'].value_counts()
