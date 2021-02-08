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

experiment_uid = Path(config['experiment_dir']).name

experiment_df = pd.read_csv(Path(os.getcwd()) / 'data/experiments_dataframe.csv')

# experiment_df = pd.read_csv(os.path.expanduser(config['experiment_df_path']))

lr_eval_columns = [col for col in experiment_df.columns if col.startswith('lr_eval_')]

best_row = experiment_df.loc[experiment_df['experiment_uid'] == experiment_uid]

print(best_row)
adfsg

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
