# HK, 27.01.21
import os
from pathlib import Path

CLF_RESULTS_PATH = Path(os.getcwd()) / 'data/clfs/clf_test_results.json'


def bold_max_value(df, df_tex: str):
    df_max = max(df.max().values.tolist())

    return df_tex.replace(str(df_max), r'\textbf{' + str(df_max) + r'}')
