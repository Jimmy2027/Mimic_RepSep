# HK, 17.01.21
import os
from pathlib import Path
import json
from logger.logger import log


def get_config_path() -> str:
    if os.path.exists('/cluster/home/klugh/'):
        return 'leomed'
    elif os.path.exists('/mnt/data/hendrik'):
        return 'bartholin'
    else:
        return 'config'


def write_to_config(values: dict):
    config = get_config()
    for k, v in values.items():
        config[k] = v
    config_path = Path(os.getcwd()) / f'configs/{get_config_path()}.json'
    log.info(f'Writing to {values} to config {config_path}.')
    with open(config_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)


def get_config() -> dict:
    config_path = Path(os.getcwd()) / f'configs/{get_config_path()}.json'
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    return config

def set_paths(flags, config):
    flags.dir_data = os.path.expanduser(config['dir_data'])
    return flags

MODALITIES = ['PA', 'Lateral', 'text']
LABELS = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']
MOD_MAPPING = {
    'PA': 'm1',
    'Lateral': 'm2',
    'text': 'm3'
}


class Dummylogger:
    def add_scalars(self, somestring: str, *kwargs):
        pass
