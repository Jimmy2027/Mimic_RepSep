# HK, 10.02.21
import json
import pandas as pd
from pathlib import Path
import os
from itertools import combinations
from scripts.utils import bold_max_value


def df_builder(labels, exp_row, mods):
    for label in labels:
        label_row = {'MODEL': 'MoPoE', 'LABEL': label}
        label_cols = [col for col in exp_row.columns if col.startswith(f'lr_eval_{label}')]
        for col in label_cols:
            sub_mods = col.replace(f'lr_eval_{label}_', '').replace('_', ',')
            for k, v in mods.items():
                sub_mods = sub_mods.replace(k, v)
            label_row[sub_mods] = exp_row[col].round(decimals=3).item()

        yield label_row


def print_lr_table(bin_labels: bool):
    config_path = Path(os.getcwd()) / 'configs/bartholin.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    if bin_labels:
        labels = ['Finding']
        experiment_uid = Path(config['experiment_dir_bin']).name

    else:
        labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
        experiment_uid = Path(config['experiment_dir']).name

    experiment_df = pd.read_csv(Path(os.getcwd()) / 'data/experiments_dataframe.csv')

    exp_row = experiment_df.loc[experiment_df['experiment_uid'] == experiment_uid]

    assert not exp_row.empty, f'{experiment_uid} is not found in experiment_df'

    mods = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}

    subsets = []
    for L in range(1, len(mods) + 1):
        for subset in combinations(mods, L):
            subsets.append(''.join(subset))

    df = pd.DataFrame(df_builder(labels, exp_row, mods))

    df.set_index(['MODEL', 'LABEL'], inplace=True)
    df.sort_index(inplace=True)

    df_tex = df.to_latex(escape=False)
    df_tex = df_tex.replace(r'\toprule', '')
    # df_tex = df_tex.replace(r'\bottomrule', '')
    print(bold_max_value(df, df_tex))
