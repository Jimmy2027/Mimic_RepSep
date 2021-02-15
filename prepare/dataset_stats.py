# HK, 11.02.21
import json
import os
from pathlib import Path

import pandas as pd
from mimic.dataio.utils import filter_labels

from logger.logger import log

classes = ['Finding']
train_df = pd.read_csv(Path(os.getcwd()) / 'data/train_labels.csv')
df = filter_labels(labels=train_df, undersample_dataset=False, split='train', which_labels=classes)

stats = {'Finding': int(df[df[classes] == 1].count().Finding),
         'NoFinding': int(df[df[classes] == 0].count().Finding)}

out_path = Path(os.getcwd()) / 'data/dataset_stats.json'
log.info(f'Saving dataset stats to {out_path}')

with open(out_path, 'w') as outfile:
    json.dump(stats, outfile)
