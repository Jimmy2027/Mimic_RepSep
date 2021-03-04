# HK, 13.02.21
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.evaluation.eval_metrics.coherence import test_generation, classify_generated_samples
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import set_paths
from mimic.networks.classifiers.utils import Metrics, get_labels
from logger.logger import log
from utils import Dummylogger
from utils import get_config
from torch.utils.data import DataLoader


def test_vae_gen():
    config = get_config()
    # set seed
    SEED = config['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # experiment_dir = config['experiment_dir']
    experiment_dir = 'binary_labels-True_beta-0.01_weighted_sampler-False_class_dim-128_text_gen_lastlayer-softmax_2021_02_10_14_56_27_974859'

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

    d_loader = DataLoader(test_set,
                          batch_size=FLAGS.batch_size,
                          shuffle=False,
                          num_workers=FLAGS.dataloader_workers, drop_last=False)
    mm_vae = mimic_experiment.mm_vae
    mods = mimic_experiment.modalities
    subsets = mimic_experiment.subsets
    if '' in subsets:
        del subsets['']

    with torch.no_grad():
        batch_labels, gen_perf, cond_gen_classified = classify_generated_samples(FLAGS, d_loader, mimic_experiment,
                                                                                 mm_vae,
                                                                                 mods, subsets)

        gen_perf_cond = {}
        # compare the classification on the generated samples with the ground truth
        for l_idx, l_key in enumerate(mimic_experiment.labels):
            gen_perf_cond[l_key] = {}
            for s_key in subsets:
                gen_perf_cond[l_key][s_key] = {}
                for m_key in mods:
                    metrics = Metrics(cond_gen_classified[s_key][m_key], batch_labels,
                                      str_labels=get_labels(FLAGS.binary_labels))
                    gen_perf_cond[l_key][s_key][m_key] = metrics.evaluate()[config['eval_metric']][0]

            eval_score = mimic_experiment.mean_eval_metric(gen_perf['random'][l_key])
            gen_perf['random'][l_key] = eval_score

        gen_perf['cond'] = gen_perf_cond

    results = gen_perf

    log.info(f'Gen eval results: {results}')

    out_path = Path(os.getcwd()) / 'data/gen_eval_results.json'
    log.info(f'Saving gen eval test results to {out_path}')
    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    test_vae_gen()
