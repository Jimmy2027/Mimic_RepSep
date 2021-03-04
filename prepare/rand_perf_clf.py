# HK, 03.03.21
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.utils.filehandling import expand_paths
from mimic.utils.flags import update_flags_with_config
from sklearn import metrics
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from logger.logger import log
from utils import LABELS, MODALITIES
from utils import get_config_path as get_config_str, get_config


def translate(batch, list_labels: list) -> list:
    """
    Translates batch label tensor to list of character labels
    """
    for elem in batch:
        if elem[0] == 1:
            list_labels.append('Lung Opacity')
        elif elem[1] == 1:
            list_labels.append('Pleural Effusion')
        elif elem[2] == 1:
            list_labels.append('Support Devices')
        else:
            list_labels.append('None')
    return list_labels


def test_clfs(flags, img_size: int, text_encoding: str):
    flags.img_size = img_size
    flags.text_encoding = text_encoding
    # set clf_training to true to get img transformations from dataset. For the same reason set flags.modality to PA
    flags.modality = 'PA'
    mimic_test = Mimic(flags, LABELS, split='eval')
    flags.batch_size = len(mimic_test)

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    results = {modality: {
        'list_precision_vals': [],
        'list_prediction_vals': [],
        'list_gt_vals': [],
    } for modality in MODALITIES}

    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(LABELS))))

        for modality, value in results.items():
            clf_input = Variable(batch_d[modality]).to(flags.device)
            prediction = np.random.randint(0, 2, size=(clf_input.shape[0], 3))
            results[modality]['list_prediction_vals'] = translate(prediction, results[modality]['list_prediction_vals'])
            results[modality]['list_gt_vals'] = translate(batch_l.cpu(), results[modality]['list_gt_vals'])
            prediction = prediction.ravel()
            avg_precision = average_precision_score(labels.ravel(), prediction)

            if not np.isnan(avg_precision):
                results[modality]['list_precision_vals'].append(avg_precision)
            else:
                warnings.warn(
                    f'avg_precision_{modality} has value {avg_precision} with labels: {labels.ravel()} and '
                    f'prediction: {prediction}')

    for modality in results:
        results[modality]['report'] = metrics.classification_report(results[modality]['list_gt_vals'],
                                                                    results[modality]['list_prediction_vals'], digits=4,
                                                                    output_dict=True)
    return results


if __name__ == '__main__':
    log.info('Starting classifier testing.')
    config = get_config()
    mimic_config_path = Path(os.getcwd()) / f'prepare/mimic_configs/{get_config_str()}.json'

    FLAGS = update_flags_with_config(mimic_config_path)
    FLAGS.dir_clf = Path(os.getcwd()) / 'data/clfs/trained_classifiers_final'
    FLAGS.reduce_lr_on_plateau = True
    FLAGS.fixed_extractor = True
    FLAGS.normalization = False
    FLAGS = expand_paths(FLAGS)
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')

    results = test_clfs(FLAGS, 128, 'word')
    for mod in results:
        print(results[mod]['list_precision_vals'])
    # out_path = f'{FLAGS.dir_clf}/clf_test_results_rand.json'
    # log.info(f'Saving classifier test results to {out_path}')
    # with open(out_path, 'w') as outfile:
    #     json.dump(results, outfile)
