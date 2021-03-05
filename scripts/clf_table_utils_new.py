# HK, 16.02.21
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from prepare.utils import get_config
from scripts.utils import bold_max_value
from typing import Union

MODALITIES = ['PA', 'Lateral', 'text']
MOD_MAPPING = {
    'PA': 'm1',
    'Lateral': 'm2',
    'text': 'm3'
}
mod_to_modsymbol = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}


@dataclass
class Params():
    dir_clf: Path = Path(os.getcwd()) / 'data/clfs/trained_classifiers_final'
    img_size: int = 128
    img_clf_type: str = 'resnet'
    bin_labels: bool = False
    vocab_size: int = 2900


def df_builder(clf_eval_results, metrics: list):
    if metrics == 'all':
        metrics = list(clf_eval_results['PA'].keys())
        metrics.remove('mean_AP_total')
        metrics.remove('f1')
    for metric in metrics:
        label_row = {'Metric': metric.replace('mean_AP_Finding', 'mean AP').replace('_', r'\_')}
        for k, v in clf_eval_results.items():
            if k != 'random_perf':
                label_row[mod_to_modsymbol[k]] = np.round(v[metric][0], 3)

        yield label_row


def print_clf_table(bin_labels: bool, metrics: list):
    config = get_config()

    if bin_labels:
        labels = ['Finding']
    else:
        labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']

    params = Params()
    params.bin_labels = bin_labels

    dir_clf = Path(os.getcwd()) / f'data/clfs/{config["dir_clf"]}'

    clf_eval_results_path = dir_clf / f'clf_test_results{"_bin_label" if bin_labels else ""}.json'

    with open(clf_eval_results_path, 'r') as json_file:
        clf_eval_results = json.load(json_file)

    mods = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}

    subsets = []
    for L in range(1, len(mods) + 1):
        for subset in combinations(mods, L):
            subsets.append(''.join(subset))

    df = pd.DataFrame(df_builder(clf_eval_results, metrics))
    df = df.reset_index(drop=True)
    df_tex = df.to_latex(index=False, escape=False)
    df_tex = df_tex.replace(r'\toprule', '')
    df_tex = df_tex.replace(r'\bottomrule', '')
    print(bold_max_value(df, df_tex))
    # print(df_tex)


if __name__ == '__main__':
    print_clf_table(bin_labels=True)
