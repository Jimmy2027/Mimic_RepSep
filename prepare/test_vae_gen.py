# HK, 13.02.21
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.evaluation.eval_metrics.coherence import test_generation
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import set_paths

from logger.logger import log
from utils import Dummylogger
from utils import get_config


def test_vae_gen():
    config = get_config()
    # set seed
    SEED = config['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    experiment_dir = config['experiment_dir']
    experiment_path = Path(os.getcwd()) / f'data/vae_model/{experiment_dir}'
    flags_path = experiment_path / 'flags.rar'
    FLAGS = torch.load(flags_path)
    FLAGS.save_figure = False
    FLAGS.dir_cond_gen = Path(__file__).parent.parent / 'data/cond_gen'
    FLAGS.text_gen_lastlayer = 'softmax'

    FLAGS = set_paths(FLAGS)
    FLAGS.dir_clf = Path(os.getcwd()) / 'data/clfs/trained_classifiers_final'
    FLAGS.dir_gen_eval_fid = Path(os.getcwd()) / 'data/gen_eval_fid'
    FLAGS.use_clf = True
    FLAGS.batch_size = 30
    state_dict_path = experiment_path / 'checkpoints/0149/mm_vae'
    FLAGS.binary_labels = True
    mimic_experiment = MimicExperiment(flags=FLAGS)
    mimic_experiment.tb_logger = Dummylogger()
    mimic_experiment.mm_vae.to(FLAGS.device)
    mimic_experiment.mm_vae.load_state_dict(state_dict=torch.load(state_dict_path))
    mimic_experiment.mm_vae.eval()
    test_set = Mimic(FLAGS, mimic_experiment.labels, split='test')
    with torch.no_grad():
        gen_eval = test_generation(0, mimic_experiment, dataset=test_set)
    results = gen_eval

    log.info(f'Gen eval results: {results}')

    out_path = Path(os.getcwd()) / 'data/gen_eval_results.json'
    log.info(f'Saving gen eval test results to {out_path}')
    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    test_vae_gen()
