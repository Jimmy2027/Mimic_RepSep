# HK, 27.01.21
import os
from pathlib import Path
from prepare.utils import get_config
import json

CLF_RESULTS_PATH = Path(os.getcwd()) / 'data/clfs/clf_test_results.json'


def bold_max_value(df, df_tex: str):
    for df_max in df.max(numeric_only=True, axis=1).values.tolist():
        df_tex = df_tex.replace(str(df_max), r'\textbf{' + str(df_max) + r'}')

    return df_tex


def get_random_perf():
    config = get_config()
    dir_clf = Path(__file__).parent.parent / f'data/clfs/{config["dir_clf"]}'
    clf_result_path = f'{dir_clf}/clf_test_results_bin_label.json'
    with open(clf_result_path, 'r') as jsonfile:
        clf_results = json.load(jsonfile)
    return clf_results['random_perf'][config['eval_metric']][0]


if __name__ == '__main__':
    print(get_random_perf())
