# HK, 17.01.21
import os
import glob
from pathlib import Path

from mimic.networks.classifiers.main_train_clf_mimic import run_training_procedure_clf
from mimic.utils.flags import update_flags_with_config

from utils import get_config_path, get_config

config = get_config()
mimic_config_path = Path(os.getcwd()) / f'prepare/mimic_configs/{get_config_path()}.json'

FLAGS = update_flags_with_config(mimic_config_path)
FLAGS.reduce_lr_on_plateau = True
FLAGS.fixed_extractor = True

mod_mapping = {
    'PA': 'm1',
    'Lateral': 'm2',
    'text': 'm3'
}

for modality in ['PA', 'Lateral', 'text']:
    print(glob.glob(f"{FLAGS.dir_clf}/clf_{mod_mapping[modality]}*"))
    if not glob.glob(f"{FLAGS.dir_clf}/clf_{mod_mapping[modality]}*"):
        FLAGS.modality = modality
        exp_uid = run_training_procedure_clf(FLAGS)
