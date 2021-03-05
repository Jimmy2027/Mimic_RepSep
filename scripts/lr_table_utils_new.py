# HK, 16.02.21

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils import bold_max_value
from prepare.utils import get_config


def df_builder(lr_eval_results, mods, metrics: str):
    if metrics == 'all':
        metrics = list(lr_eval_results['PA'].keys())
        metrics.remove('mean_AP_total')
        metrics.remove('f1')
    for metric in metrics:
        label_row = {'Metric': metric.replace('mean_AP_Finding', 'mean AP').replace('_', r'\_')}
        for col, scores in lr_eval_results.items():
            sub_mods = col.replace('_', ',')
            for k, v in mods.items():
                sub_mods = sub_mods.replace(k, v)
            label_row[sub_mods] = round(scores[metric][0], 3)
        yield label_row


def print_lr_table(bin_labels: bool, metrics):
    if bin_labels:
        labels = ['Finding']

    else:
        labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']

    lr_eval_results_path = Path(os.getcwd()) / f'data/lr_eval_results{"_bin_label" if bin_labels else ""}.json'

    with open(lr_eval_results_path, 'r') as json_file:
        lr_eval_results = json.load(json_file)

    mods = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}

    subsets = []
    for L in range(1, len(mods) + 1):
        for subset in combinations(mods, L):
            subsets.append(''.join(subset))
    config = get_config()

    df = pd.DataFrame(df_builder(lr_eval_results, mods, metrics))

    # df.set_index(['MODEL', 'LABEL'], inplace=True)
    # df.sort_index(inplace=True)
    #
    # df_tex = df.to_latex(escape=False)

    df = df.reset_index(drop=True)
    df_tex = df.to_latex(index=False, escape=False)
    df_tex = df_tex.replace(r'\toprule', '')
    df_tex = df_tex.replace(r'\bottomrule', '')
    print(bold_max_value(df, df_tex))
    # print(df_tex)
