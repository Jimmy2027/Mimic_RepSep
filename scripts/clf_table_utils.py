# HK, 10.02.21
import glob
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from scripts.utils import bold_max_value

import pandas as pd
import torch

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


def df_builder(labels, experiment_df, experiment_uids, mods):
    exp_rows = {mod: experiment_df.loc[experiment_df['experiment_uid'] == experiment_uids[mod]] for mod in mods}
    for metric in ['mean_AP_Finding', 'specificity', 'accuracy']:
        label_row = {'MODEL': 'ResNet', 'Metric': metric}
        for mod in mods:
            label_row[mod_to_modsymbol[mod]] = exp_rows[mod][metric].round(decimals=3).item()

        yield label_row


def get_clf_uids(args: Params) -> dict:
    clf_uids = {}
    for modality in MODALITIES:
        if modality in ['PA', 'Lateral']:
            dir_clf = f'{args.dir_clf}/Mimic{args.img_size}_{args.img_clf_type}{"_bin_label" if args.bin_labels else ""}'
            flags_path = Path(glob.glob(f"{dir_clf}/flags_clf_{modality}*")[0])

        elif modality == 'text':
            dir_clf = args.dir_clf

            if args.bin_labels:
                flags_path = [f for f in glob.glob(f'{dir_clf}/flags_clf_{modality}vocabsize_{args.vocab_size}*') if
                              'bin_label' in f][0]
            else:
                flags_path = [f for f in glob.glob(f'{dir_clf}/flags_clf_{modality}vocabsize_{args.vocab_size}*') if
                              'bin_label' not in f][0]

        else:
            raise NotImplementedError

        flags = torch.load(flags_path)

        clf_uids[modality] = flags.experiment_uid

    return clf_uids


def print_clf_table(bin_labels: bool):
    if bin_labels:
        labels = ['Finding']
    else:
        labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']

    params = Params()
    params.bin_labels = bin_labels
    experiment_uids = get_clf_uids(params)

    experiment_df = pd.read_csv(Path(os.getcwd()) / 'data/clf_experiments_dataframe.csv')

    mods = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}

    subsets = []
    for L in range(1, len(mods) + 1):
        for subset in combinations(mods, L):
            subsets.append(''.join(subset))

    df = pd.DataFrame(df_builder(labels, experiment_df, experiment_uids, mods))
    df.set_index(['MODEL'], inplace=True)
    df.sort_index(inplace=True)

    df_tex = df.to_latex(escape=False)
    print(bold_max_value(df, df_tex))
    print(df_tex)

if __name__ == '__main__':
    print_clf_table(bin_labels=True)
