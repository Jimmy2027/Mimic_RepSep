# HK, 17.01.21
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg as ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText as ClfText
from mimic.utils.filehandling import expand_paths
from mimic.utils.flags import update_flags_with_config
from sklearn import metrics
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from logger.logger import log
from utils import MOD_MAPPING, LABELS, MODALITIES
from utils import get_config_path as get_config_str, get_config


def translate(batch, list_labels: list) -> list:
    """
    Translates batch label tensor to list of character labels
    """
    for elem in batch:
        elem = elem.detach().numpy()
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
    mimic_test = Mimic(flags, LABELS, split='eval', clf_training=True)
    flags.batch_size = len(mimic_test)
    clfs = load_clfs(FLAGS)

    models = {}

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    results = {}
    for modality in MODALITIES:
        models[modality] = clfs[modality].eval()
        results[modality] = {
            'list_precision_vals': [],
            'list_prediction_vals': [],
            'list_gt_vals': [],
        }

    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(LABELS))))

        for modality in results:
            clf_input = Variable(batch_d[modality]).to(flags.device)
            prediction = models[modality](clf_input).cpu()
            results[modality]['list_prediction_vals'] = translate(prediction, results[modality]['list_prediction_vals'])
            results[modality]['list_gt_vals'] = translate(batch_l.cpu(), results[modality]['list_gt_vals'])
            prediction = prediction.data.numpy().ravel()
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


def load_clfs(args) -> dict:
    dir_clf = f'{FLAGS.dir_clf}/Mimic{FLAGS.img_size}_{FLAGS.img_clf_type}'
    clfs = {}
    for modality in MODALITIES:
        print(f"{dir_clf}/clf_{MOD_MAPPING[modality]}*")
        print(glob.glob(f"{dir_clf}/clf_{MOD_MAPPING[modality]}*"))
        clf_path = glob.glob(f"{dir_clf}/clf_{MOD_MAPPING[modality]}*")[0]
        if modality in ['PA', 'Lateral']:
            clf = ClfImg(args, LABELS) if args.img_clf_type == 'resnet' else CheXNet(
                len(LABELS))
        elif modality == 'text':
            clf = ClfText(args, LABELS)
        else:
            raise NotImplementedError

        clf.load_state_dict(torch.load(clf_path))
        clfs[modality] = clf.to(args.device)

    return clfs


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
    out_path = f'{FLAGS.dir_clf}/clf_test_results.json'
    log.info(f'Saving classifier test results to {out_path}')
    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)
