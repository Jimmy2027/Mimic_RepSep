import json

import pandas as pd

from scripts.utils import CLF_RESULTS_PATH

clf_results_path = CLF_RESULTS_PATH

with open(clf_results_path, 'r') as json_file:
    clf_results = json.load(json_file)
mods_mapping = {'PA': 'F', 'Lateral': 'L', 'text': 'T'}

table_dict = {
    mods_mapping[mod]: {'mean AP': clf_results[mod]['list_precision_vals'][0]}
    for mod in ['PA', 'Lateral', 'text']
}

df = pd.DataFrame(table_dict)
df_tex = df.to_latex(escape=False)
print(df_tex)
