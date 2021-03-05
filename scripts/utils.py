# HK, 27.01.21
import os
from pathlib import Path
from prepare.utils import get_config
import json

CLF_RESULTS_PATH = Path(os.getcwd()) / 'data/clfs/clf_test_results.json'


def bold_max_value(df, df_tex: str):
    """
    Wraps the maximum value for each row with \textbf{}.

    Parameters
    ----------
    df :  pandas.DataFrame
            Pandas DataFrame object containing at least one column with numerical values.

    df_tex: string
                The output of df.to_latex(). The values if this string will be changed such that the maximum values of
                each row will be bold in the resulting pdf.
    >>> import pandas as pd
    >>> df = pd.DataFrame({'col1': [1, 2, 20], 'col2': [3, 4, 8],'col3': [5, 6, 5]})
    >>> print(bold_max_value(df, df.to_latex()), end="")
    \\begin{tabular}{lrrr}
    \\toprule
    {} &  col1 &  col2 &  col3 \\\\
    \\midrule
    0 &     1 &     3 &     \\textbf{5} \\\\
    1 &     2 &     4 &     \\textbf{6} \\\\
    2 &    \\textbf{20} &     8 &     5 \\\\
    \\bottomrule
    \\end{tabular}
    """
    header = df_tex.find(r'\midrule') + len('\midrule\n')
    tail = df_tex.find('\n\\bottomrule')
    body = df_tex[header:tail].split('\n')
    for row_idx, (row_max, tex_row) in enumerate(zip(df.max(numeric_only=True, axis=1).values.tolist(), body)):
        body[row_idx] = tex_row.replace(str(row_max), r'\textbf{' + str(row_max) + r'}')

    return df_tex[:header] + '\n'.join(body) + df_tex[tail:]


def get_random_perf():
    config = get_config()
    dir_clf = Path(__file__).parent.parent / f'data/clfs/{config["dir_clf"]}'
    clf_result_path = f'{dir_clf}/clf_test_results_bin_label.json'
    with open(clf_result_path, 'r') as jsonfile:
        clf_results = json.load(jsonfile)
    return clf_results['random_perf'][config['eval_metric']][0]


if __name__ == '__main__':
    # import pandas as pd

    import doctest

    doctest.testmod()
    print(get_random_perf())
