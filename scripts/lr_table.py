# HK, 16.01.21
import json
import pandas as pd
from pathlib import Path

config_path = Path(__file__).parent.parent / 'configs/bartholin.json'

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

experiment_df = pd.read_csv(config['experiment_df_path'])


