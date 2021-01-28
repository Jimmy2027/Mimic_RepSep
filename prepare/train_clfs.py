# HK, 17.01.21
import glob
import os
from pathlib import Path

from mimic.networks.classifiers.main_train_clf_mimic import run_training_procedure_clf
from mimic.utils.flags import update_flags_with_config

from logger.logger import log
from utils import get_config_path, MOD_MAPPING

mimic_config_path = Path(os.getcwd()) / f'prepare/mimic_configs/{get_config_path()}.json'

FLAGS = update_flags_with_config(mimic_config_path)
FLAGS.reduce_lr_on_plateau = True
FLAGS.fixed_extractor = True

if FLAGS.dir_clf != 'text':
    dir_clf = f'{FLAGS.dir_clf}/Mimic{FLAGS.img_size}_{FLAGS.img_clf_type}'
else:
    dir_clf = FLAGS.dir_clf
for modality in ['PA', 'Lateral', 'text']:
    print(glob.glob(f"{dir_clf}/clf_{MOD_MAPPING[modality]}*"))
    if not glob.glob(f"{dir_clf}/clf_{MOD_MAPPING[modality]}*"):
        FLAGS.modality = modality
        exp_uid = run_training_procedure_clf(FLAGS)
    else:
        log.info(f'Found {modality} classifier. Training skipped.')
