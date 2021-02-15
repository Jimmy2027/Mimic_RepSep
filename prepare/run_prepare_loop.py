# HK, 13.02.21
import json
import os
from pathlib import Path

import pandas as pd

from logger.logger import log
from make_cond_gen_fig import make_cond_gen_fig
from test_vae_gen import test_vae_gen


# run with /home/hendrik/miniconda3/envs/mimic/bin/python prepare/run_prepare_loop.py

parent_dir = Path('/mnt/data/hendrik/mimic_scratch/mimic/moe/test_beta_bigsearch')
config_path = Path(os.getcwd()) / 'configs/bartholin.json'

experiment_df = pd.read_csv(Path(os.getcwd()) / 'data/experiments_dataframe.csv')

for experiment_dir in parent_dir.iterdir():

    experiment_uid = experiment_dir.name
    if experiment_uid in experiment_df['experiment_uid'].tolist():

        dest_dir = Path(os.getcwd()) / f'data/vae_model/{experiment_uid}'
        if not dest_dir.exists():
            symlink_command = f'ln -s {experiment_dir} {dest_dir}'
            log.info(f'Running {symlink_command}')
            os.system(symlink_command)
        with open(config_path, 'r') as json_file:
            config = json.load(json_file)
        config['experiment_dir'] = experiment_uid
        config['experiment_dir_bin'] = experiment_uid

        with open(config_path, 'w') as json_file:
            json.dump(config, json_file)

        make_cond_gen_fig()
        test_vae_gen()

        os.system('./prepare/run_loop.sh')
    else:
        print(f'{experiment_uid} is not found in experiment_df')
