# HK, 12.02.21
import json
import os
import random
import typing
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.evaluation.eval_metrics.representation import train_clf_lr_all_subsets, classify_latent_representations
from mimic.networks.classifiers.utils import Metrics, get_labels
from mimic.utils.experiment import MimicExperiment
from mimic.utils.utils import dict_to_device
from torch.utils.data import DataLoader

from logger.logger import log
from utils import Dummylogger
from utils import get_config
from utils import set_paths


def test_clf_lr_all_subsets(clf_lr, exp):
    """
    Test the classifiers that were trained on latent representations.
    """
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    subsets = exp.subsets
    if '' in subsets:
        del subsets['']

    test_set = Mimic(args, exp.labels, split='test')

    d_loader = DataLoader(test_set, batch_size=exp.flags.batch_size, shuffle=False, num_workers=0, drop_last=False)

    if exp.flags.steps_per_training_epoch > 0:
        training_steps = exp.flags.steps_per_training_epoch
    else:
        training_steps = len(d_loader)
    log.info(f'Creating {training_steps} batches of latent representations for classifier testing '
             f'with a batch_size of {exp.flags.batch_size}.')

    clf_predictions = {subset: torch.Tensor() for subset in subsets}

    batch_labels = torch.Tensor()

    for iteration, (batch_d, batch_l) in enumerate(d_loader):
        if iteration > training_steps:
            break
        batch_labels = torch.cat((batch_labels, batch_l), 0)

        batch_d = dict_to_device(batch_d, exp.flags.device)

        inferred = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        lr_subsets = inferred['subsets']
        data_test = {key: lr_subsets[key][0].cpu().data.numpy() for key in lr_subsets}

        clf_predictions_batch = classify_latent_representations(exp, clf_lr, data_test)
        clf_predictions_batch: Mapping[str, Mapping[str, np.array]]

        for subset in subsets:
            clf_predictions_batch_subset = torch.cat(tuple(
                torch.tensor(clf_predictions_batch[label][subset]).unsqueeze(1) for label in
                get_labels(args.binary_labels)), 1)

            clf_predictions[subset] = torch.cat([clf_predictions[subset], clf_predictions_batch_subset], 0)

    return clf_predictions, batch_labels


def eval_vae_lr():
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
    FLAGS.save_figure = True
    FLAGS.dir_cond_gen = Path(__file__).parent.parent / 'data/cond_gen'
    FLAGS.text_gen_lastlayer = 'softmax'

    FLAGS = set_paths(FLAGS, config)
    FLAGS.use_clf = False
    FLAGS.batch_size = 30
    # FLAGS.undersample_dataset = True
    state_dict_path = experiment_path / 'checkpoints/0149/mm_vae'

    mimic_experiment = MimicExperiment(flags=FLAGS)
    mimic_experiment.tb_logger = Dummylogger()
    mimic_experiment.mm_vae.to(FLAGS.device)
    mimic_experiment.mm_vae.load_state_dict(state_dict=torch.load(state_dict_path))
    mimic_experiment.mm_vae.eval()

    results = {}
    for binay_labels in [True]:
        mimic_experiment.flags.binary_labels = binay_labels
        with torch.no_grad():
            clf_lr = train_clf_lr_all_subsets(mimic_experiment, weighted_sampler=False)
            predictions, gt = test_clf_lr_all_subsets(clf_lr, mimic_experiment)
            for subset in predictions:
                # calculate metrics
                metrics = Metrics(predictions[subset], gt, str_labels=get_labels(FLAGS.binary_labels))
                metrics_dict = metrics.evaluate()
                results[subset] = metrics_dict
                print(subset, ':', metrics_dict[config['eval_metric']][0])

    log.info(f'Lr eval results: {results}')

    out_path = Path(os.getcwd()) / f'data/lr_eval_results{"_bin_label" if binay_labels else ""}.json'
    log.info(f'Saving lr eval test results to {out_path}')

    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    eval_vae_lr()
