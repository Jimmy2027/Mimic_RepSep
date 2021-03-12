# HK, 12.02.21
import glob
import json
import os
from pathlib import Path

import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.networks.CheXNet import CheXNet
from mimic.networks.ConvNetworkImgClf import ClfImg as ClfImg
from mimic.networks.ConvNetworkTextClf import ClfText as ClfText
from mimic.networks.classifiers.main_train_clf_mimic import eval_clf
from mimic.networks.classifiers.utils import Metrics, get_labels
from mimic.utils.filehandling import expand_paths
from mimic.utils.flags import update_flags_with_config
from mimic.utils.loss import get_clf_loss
from torch.utils.data import DataLoader

from logger.logger import log
from utils import MOD_MAPPING, MODALITIES, Dummylogger
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
    mimic_test = Mimic(flags, get_labels(flags.binary_labels), split='test')
    flags.batch_size = len(mimic_test)
    clfs = load_clfs(flags)

    models = {}

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0,
                                             drop_last=False)
    results = {}
    for modality in MODALITIES:
        models[modality] = clfs[modality].eval()
        results[modality] = {
            'list_precision_vals': [],
            'list_prediction_vals': [],
            'list_gt_vals': [],
        }
    criterion = get_clf_loss(flags.clf_loss)
    dummylogger = Dummylogger()

    results = {}
    for mod in [*MODALITIES, 'random_perf']:
        if mod != 'random_perf':
            loss, val_results = eval_clf(flags, epoch=0, model=models[mod], data_loader=dataloader,
                                         log_writer=dummylogger,
                                         modality=mod, criterion=criterion)
        else:
            val_results['predictions'] = torch.randint(0, 2, val_results['ground_truths'].shape)
        # calculate metrics
        metrics = Metrics(val_results['predictions'], val_results['ground_truths'],
                          str_labels=get_labels(flags.binary_labels))
        metrics_dict = metrics.evaluate()
        print(mod)
        print(metrics_dict)
        results[mod] = metrics_dict

    return results


def load_clfs(args) -> dict:
    clfs = {}
    for modality in MODALITIES:
        log.info(f'Loading {modality} clf.')
        if modality in ['PA', 'Lateral']:
            clf = ClfImg(args, get_labels(args.binary_labels)) if args.img_clf_type == 'resnet' else CheXNet(
                len(get_labels(args.binary_labels)))

            dir_clf = f'{args.dir_clf}/Mimic{args.img_size}_{args.img_clf_type}{"_bin_label" if args.binary_labels else ""}'
            clf_path = Path(glob.glob(f"{dir_clf}/clf_{MOD_MAPPING[modality]}*")[0])

        elif modality == 'text':
            dir_clf = args.dir_clf
            clf = ClfText(args, get_labels(args.binary_labels))
            if args.binary_labels:
                clf_path = \
                    [f for f in glob.glob(f'{dir_clf}/clf_{MOD_MAPPING[modality]}vocabsize_{args.vocab_size}*') if
                     'bin_label' in f][0]
            else:
                clf_path = \
                    [f for f in glob.glob(f'{dir_clf}/clf_{MOD_MAPPING[modality]}vocabsize_{args.vocab_size}*') if
                     'bin_label' not in f][0]

        else:
            raise NotImplementedError
        log.info(f'Loading state dict from {clf_path}.')
        clf.load_state_dict(torch.load(clf_path))
        clfs[modality] = clf.to(args.device)

    return clfs


if __name__ == '__main__':
    log.info('Starting classifier testing.')
    config = get_config()
    mimic_config_path = Path(os.getcwd()) / f'prepare/mimic_configs/{get_config_str()}.json'

    FLAGS = update_flags_with_config(mimic_config_path)
    FLAGS.dir_clf = Path(os.getcwd()) / f'data/clfs/{config["dir_clf"]}'
    FLAGS.reduce_lr_on_plateau = True
    FLAGS.fixed_extractor = True
    FLAGS.normalization = False
    FLAGS = expand_paths(FLAGS)
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
    FLAGS.binary_labels = True
    FLAGS.img_clf_type = 'resnet'

    results = test_clfs(FLAGS, 128, 'word')

    out_path = f'{FLAGS.dir_clf}/clf_test_results{"_bin_label" if FLAGS.binary_labels else ""}.json'
    log.info(f'Saving classifier test results to {out_path}')
    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)
