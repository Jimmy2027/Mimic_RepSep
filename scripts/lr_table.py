# HK, 16.01.21
import json
import pandas as pd
from pathlib import Path
import os
from itertools import combinations

config_path = Path(os.getcwd()) / 'configs/bartholin.json'
labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

experiment_df = pd.read_csv(os.path.expanduser(config['experiment_df_path']))

lr_eval_columns = [col for col in experiment_df.columns if col.startswith('lr_eval_')]

experiment_df = experiment_df.dropna(subset=[*lr_eval_columns])
lr_eval_avg = experiment_df[lr_eval_columns].apply(pd.DataFrame.describe, axis=1)
best_idx = lr_eval_avg[['mean']].idxmax().item()

# iloc uses integer location and does not correspond to true index:
# https://stackoverflow.com/questions/49960597/pandas-using-iloc-to-retrieve-data-does-not-match-input-index
best_row = experiment_df.loc[best_idx]

mods = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}

subsets = []
for L in range(1, len(mods) + 1):
    for subset in combinations(mods, L):
        subsets.append(''.join(subset))


def df_builder():
    for label in labels:
        label_row = {'MODEL': 'MoPoE', 'LABEL': label}
        label_cols = [col for col in experiment_df.columns if col.startswith(f'lr_eval_{label}')]
        for col in label_cols:
            sub_mods = col.replace(f'lr_eval_{label}_', '').replace('_', ',')
            for k, v in mods.items():
                sub_mods = sub_mods.replace(k, v)
            label_row[sub_mods] = best_row[col].round(decimals=3)
        yield label_row


df = pd.DataFrame(df_builder())

df.set_index(['MODEL', 'LABEL'], inplace=True)
df.sort_index(inplace=True)

# df = df.reset_index(drop=True)
# df_tex = df.to_latex(index=False, escape=False)
df_tex = df.to_latex(escape=False)
print(df_tex)
