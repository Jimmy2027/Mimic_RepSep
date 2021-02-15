# HK, 12.02.21
# HK, 17.01.21
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
    mimic_test = Mimic(flags, get_labels(flags.binary_labels), split='eval')
    flags.batch_size = len(mimic_test)
    clfs = load_clfs(flags)

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
    criterion = get_clf_loss(flags.clf_loss)
    dummylogger = Dummylogger()

    for mod in MODALITIES:
        loss, val_results = eval_clf(flags, epoch=0, model=models[mod], data_loader=dataloader, log_writer=dummylogger,
                                     modality=mod, criterion=criterion)
        # calculate metrics
        metrics = Metrics(val_results['predictions'], val_results['ground_truths'],
                          str_labels=get_labels(flags.binary_labels))
        metrics_dict = metrics.evaluate()
        print(metrics_dict)

    # for idx, batch in enumerate(dataloader):
    #     batch_d = batch[0]
    #     batch_l = batch[1]
    #     labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(LABELS))))
    #
    #     for modality in results:
    #         clf_input = Variable(batch_d[modality]).to(flags.device)
    #         prediction = models[modality](clf_input).cpu()
    #         results[modality]['list_prediction_vals'] = translate(prediction, results[modality]['list_prediction_vals'])
    #         results[modality]['list_gt_vals'] = translate(batch_l.cpu(), results[modality]['list_gt_vals'])
    #         prediction = prediction.data.numpy().ravel()
    #         avg_precision = average_precision_score(labels.ravel(), prediction)
    #
    #         if not np.isnan(avg_precision):
    #             results[modality]['list_precision_vals'].append(avg_precision)
    #         else:
    #             warnings.warn(
    #                 f'avg_precision_{modality} has value {avg_precision} with labels: {labels.ravel()} and '
    #                 f'prediction: {prediction}')
    #
    # for modality in results:
    #     results[modality]['report'] = metrics.classification_report(results[modality]['list_gt_vals'],
    #                                                                 results[modality]['list_prediction_vals'], digits=4,
    #                                                                 output_dict=True)
    # return results


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
    FLAGS.dir_clf = Path(os.getcwd()) / 'data/clfs/trained_classifiers_final'
    FLAGS.reduce_lr_on_plateau = True
    FLAGS.fixed_extractor = True
    FLAGS.normalization = False
    FLAGS = expand_paths(FLAGS)
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
    FLAGS.binary_labels = True
    FLAGS.img_clf_type = 'resnet'

    results = test_clfs(FLAGS, 128, 'word')

    out_path = f'{FLAGS.dir_clf}/clf_test_results.json'
    log.info(f'Saving classifier test results to {out_path}')
    with open(out_path, 'w') as outfile:
        json.dump(results, outfile)
