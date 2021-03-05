# HK, 18.01.21
import json
from pathlib import Path

import numpy as np
import pandas as pd
import os
from scripts.utils import bold_max_value, get_random_perf
from prepare.utils import get_config

label = 'Finding'

config = get_config()
gen_eval_results_path = Path(os.getcwd()) / 'data/gen_eval_results.json'

with open(gen_eval_results_path, 'r') as json_file:
    gen_eval_results = json.load(json_file)

mods_mapping = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}


def df_builder(cond_values: dict):
    label_row = {'MODEL': 'MoPoE', 'Metric': config['eval_metric'].replace('_', ' ')}
    # label_row = {'MODEL': 'MoPoE', 'LABEL': label, 'rand perf': get_random_perf()}
    for k, v in cond_values.items():
        label_row[mods_mapping[k]] = np.round(v, 3)
    yield label_row


dfs = {}


def translate_mods(mods):
    return ','.join(mods_mapping[elem] for elem in mods.split('_'))


for cond_mods, cond_values in gen_eval_results['cond'][label].items():
    if translate_mods(cond_mods) in ['F', 'L,F', 'L,F,T']:
        dfs[translate_mods(cond_mods)] = pd.DataFrame(df_builder(cond_values))

for _, v in dfs.items():
    v.set_index(['MODEL', 'Metric'], inplace=True)
    # v.set_index(['MODEL', 'LABEL'], inplace=True)
df = pd.concat(dfs, axis=1)
df.sort_index(inplace=True)

# df['random'] = [best_row[f'gen_eval_random_{label}'] for label in labels]

df_tex = df.to_latex(escape=False)

# df_tex = r' \begin{sc}' + df_tex + r'\end{sc} '
# df_tex = df_tex.replace('lllrrrrrrrrr', 'lccccccccccc')
df_tex = df_tex.replace('llrrrrrrrrr', 'lcccccccccc')
df_tex = df_tex.replace(r'\multicolumn{3}{l}', r'\multicolumn{3}{c}')

start_find = r'\multicolumn{3}{c}{L,F,T} \\'
start = df_tex.find(start_find) + len(start_find)
end = df_tex.find(r'\midrule')
newline = r'  \cmidrule(l){3-5} \cmidrule(l){6-8} \cmidrule(l){9-11} MODEL &    Metric     &      F &         L &         T &         F &         L &         T &         F &         L &         T \\'
# newline = r'  \cmidrule(l){4-6} \cmidrule(l){7-9} \cmidrule(l){10-12} MODEL &    LABEL & rand perf    &      F &         L &         T &         F &         L &         T &         F &         L &         T \\'
df_tex = df_tex.replace(df_tex[start:end], newline)
df_tex = df_tex.replace(r'\toprule', '')
df_tex = df_tex.replace(r'\bottomrule', '')

print(bold_max_value(df, df_tex))
