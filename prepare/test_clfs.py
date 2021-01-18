# HK, 17.01.21
import warnings

import numpy as np
import torch
from mimic.dataio.MimicDataset import Mimic
from mimic.utils.experiment import MimicExperiment
from sklearn import metrics
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader


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
    mimic_experiment = MimicExperiment(flags=flags)

    mimic_test = Mimic(flags, mimic_experiment.labels, split='eval')

    models = {}

    dataloader = torch.utils.data.DataLoader(mimic_test, batch_size=flags.batch_size, shuffle=True, num_workers=0)
    results = {}
    for modality in ['PA', 'Lateral', 'text']:
        models[modality] = mimic_experiment.clfs[modality].eval()
        results[modality] = {
            'list_precision_vals': [],
            'list_prediction_vals': [],
            'list_gt_vals': [],
        }

    for idx, batch in enumerate(dataloader):
        batch_d = batch[0]
        batch_l = batch[1]
        labels = np.array(np.reshape(batch_l, (batch_l.shape[0], len(mimic_experiment.labels))))

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
                                                                    results[modality]['list_prediction_vals'], digits=4)
    return results
