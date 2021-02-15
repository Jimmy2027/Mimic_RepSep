# HK, 12.02.21
import json
import os
import shutil
from pathlib import Path

config_path = Path(os.getcwd()) / 'configs/bartholin.json'
with open(config_path, 'r') as json_file:
    config = json.load(json_file)

experiment_uid_bin = Path(config['experiment_dir_bin']).name

experiment_uid = Path(config['experiment_dir']).name

vae_model_dir = Path(os.getcwd()) / 'data/vae_model'

vae_bin_dir = vae_model_dir / experiment_uid_bin
vae_dir = vae_model_dir / experiment_uid

for dest_path in [vae_bin_dir, vae_dir]:
    shutil.copyfile(Path(os.getcwd()) / 'article.pdf', dest_path / 'article.pdf')
