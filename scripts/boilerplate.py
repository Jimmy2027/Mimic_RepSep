# HK, 27.01.21

import json

from utils import CLF_RESULTS_PATH


def read_rand_pref():
    clf_results_path = CLF_RESULTS_PATH

    with open(clf_results_path, 'r') as json_file:
        clf_results = json.load(json_file)
    print(clf_results['rand_perf'])


if __name__ == '__main__':
    read_rand_pref()