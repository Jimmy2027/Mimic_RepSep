# HK, 10.02.21
# from scripts.clf_table_utils import print_clf_table
from scripts.clf_table_utils_new import print_clf_table


print_clf_table(bin_labels=True, metrics=['mean_AP_Finding', 'accuracy', 'specificity'])
