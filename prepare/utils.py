# HK, 17.01.21
import os
from pathlib import Path
import json


def get_config_path() -> str:
    if os.path.exists('/cluster/home/klugh/'):
        return 'leomed'
    elif os.path.exists('/mnt/data/hendrik'):
        return 'bartholin'
    else:
        return 'local'


def get_config() -> dict:
    config_path = Path(os.getcwd()) / f'configs/{get_config_path()}.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    return config


MODALITIES = ['PA', 'Lateral', 'text']
LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
MOD_MAPPING = {
    'PA': 'm1',
    'Lateral': 'm2',
    'text': 'm3'
}
