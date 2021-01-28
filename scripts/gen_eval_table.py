# HK, 18.01.21
import json
import os

import pandas as pd

config_path = '/home/hendrik/docsrc/mimic_repsep/configs/bartholin.json'
labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']

with open(config_path, 'r') as json_file:
    config = json.load(json_file)
experiment_df = pd.read_csv(os.path.expanduser(config['experiment_df_path']))
gen_eval_columns = [col for col in experiment_df.columns if col.startswith('gen_eval_')]

experiment_df = experiment_df.dropna(subset=[*gen_eval_columns])
gen_eval_avg = experiment_df[gen_eval_columns].apply(pd.DataFrame.describe, axis=1)
best_idx = gen_eval_avg[['mean']].idxmax().item()

best_row = experiment_df.loc[best_idx]

mods_mapping = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}


def df_builder(conditioner_mod: str):
    for label in labels:
        label_row = {'MODEL': 'MoPoE', 'LABEL': label}
        label_cols = [col for col in gen_eval_columns if
                      col.startswith(f'gen_eval_cond_{label}_{conditioner_mod}')]
        for gen_eval_col in label_cols:
            split_mods = gen_eval_col.replace(f'gen_eval_cond_', '').split('_')
            sub_mods = ','.join(split_mods[2:])
            for k, v in mods_mapping.items():
                sub_mods = sub_mods.replace(k, v)
            label_row[sub_mods] = best_row[gen_eval_col].round(decimals=3)

        yield label_row


dfs = {
    'F': pd.DataFrame(df_builder('PA')),
    'T': pd.DataFrame(df_builder('text')),
    'L': pd.DataFrame(df_builder('Lateral')),
}
for _, v in dfs.items():
    v.set_index(['MODEL', 'LABEL'], inplace=True)
df = pd.concat(dfs, axis=1)
df.sort_index(inplace=True)
df['random'] = [best_row[f'gen_eval_random_{label}'] for label in labels]

df_tex = df.to_latex(escape=False)
print(df_tex)
