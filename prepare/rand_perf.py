# HK, 27.01.21
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.utils.flags import update_flags_with_config
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import get_config_path, LABELS
import json
from logger.logger import log


def test_dummy(flags, modality: str = 'PA'):
    """
    Trains and evaluates a dummy classifier on the test set as baseline.
    Returns the average precision values
    """
    log.info('Starting dummy test.')
    mimic_test = Mimic(flags, LABELS, split='eval')
    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True,
                                             num_workers=0,
                                             drop_last=True)
    list_batches = []
    list_labels = []
    list_precision_vals = []

    for idx, (batch_d, batch_l) in enumerate(dataloader):
        ground_truth = batch_l.cpu().data.numpy()
        clf_input = Variable(batch_d[modality]).cpu().data.numpy()
        list_batches.extend(clf_input)
        list_labels.extend(ground_truth)
    # dummy classifier has no partial_fit, so all the data must be fed at once
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(list_batches, list_labels)
    # test dummy clf
    for idx, (batch_d, batch_l) in enumerate(dataloader):
        clf_input = Variable(batch_d[modality]).cpu().data.numpy()
        predictions = dummy_clf.predict(clf_input)
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(LABELS)))).ravel()
        avg_precision = average_precision_score(labels, predictions.ravel())
        if not np.isnan(avg_precision):
            list_precision_vals.append(avg_precision)
        else:
            warnings.warn(
                f'avg_precision_{modality} has value {avg_precision} with labels: {labels.ravel()} and '
                f'prediction: {predictions.cpu().data.numpy().ravel()}')
    return list_precision_vals


mimic_config_path = Path(os.getcwd()) / f'prepare/mimic_configs/{get_config_path()}.json'
FLAGS = update_flags_with_config(mimic_config_path)
out_path = f'{FLAGS.dir_clf}/clf_test_results.json'

with open(out_path, 'r') as outfile:
    results = json.load(outfile)

log.info(f'Saving dummy classifier test results to {out_path}')
results = {**results, 'rand_perf': np.mean(test_dummy(FLAGS, modality='PA'))}

with open(out_path, 'w') as outfile:
    json.dump(results, outfile)
