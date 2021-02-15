# HK, 27.01.21

import contextlib
import json
import os
import sys
from pathlib import Path
from prepare.utils import get_config
from lib.utils import float_to_tex
import pandas as pd
import torch


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    """
    Taken from
    https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    save_stdout = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stdout = save_stdout


def read_rand_perf():
    clf_results_path = Path(os.getcwd()) / 'data/clfs/clf_test_results.json'

    with open(clf_results_path, 'r') as json_file:
        clf_results = json.load(json_file)
    print(clf_results['rand_perf'])


def print_finding_counts():
    dataset_stats_json = Path(os.getcwd()) / 'data/dataset_stats.json'
    with open(dataset_stats_json, 'r') as json_file:
        stats = json.load(json_file)
    return stats['Finding']


def print_nofinding_counts():
    dataset_stats_json = Path(os.getcwd()) / 'data/dataset_stats.json'
    with open(dataset_stats_json, 'r') as json_file:
        stats = json.load(json_file)
    return stats['NoFinding']


def get_model_flags():
    config = get_config()
    flags_path = Path(os.getcwd()) / f'data/vae_model/{config["experiment_dir_bin"]}/flags.rar'
    return torch.load(flags_path)


def print_beta_value():
    flags = get_model_flags()
    return str(flags.beta)


def print_class_dim():
    flags = get_model_flags()
    return str(flags.class_dim)


def print_lr():
    flags = get_model_flags()
    # print(flags.__dict__['initial_learning_rate'])
    return str(flags.initial_learning_rate)


def print_img_shape():
    flags = get_model_flags()
    return f'({flags.img_size}, {flags.img_size})'


def print_flag_attribute(which: str):
    flags = get_model_flags()
    if which == 'img_size':
        return f'({flags.img_size}, {flags.img_size})'
    return float_to_tex(flags.__dict__[which], max_len=3)


if __name__ == '__main__':
    print(print_flag_attribute('initial_learning_rate'))
    # print(print_lr())
