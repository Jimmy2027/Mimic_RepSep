# HK, 27.01.21

import contextlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lib.utils import float_to_tex
from logger.logger import log
from prepare.utils import get_config


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    """
    Taken from
    https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    save_stdout = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stdout = save_stdout


def read_rand_perf():
    clf_results_path = Path(os.getcwd()) / 'data/clfs/clf_test_results.json'

    with open(clf_results_path, 'r') as json_file:
        clf_results = json.load(json_file)
    print(clf_results['rand_perf'])


def print_finding_counts():
    dataset_stats_json = Path(os.getcwd()) / 'data/dataset_stats.json'
    with open(dataset_stats_json, 'r') as json_file:
        stats = json.load(json_file)
    return stats['Finding']


def print_nofinding_counts():
    dataset_stats_json = Path(os.getcwd()) / 'data/dataset_stats.json'
    with open(dataset_stats_json, 'r') as json_file:
        stats = json.load(json_file)
    return stats['NoFinding']


def get_model_flags():
    config = get_config()
    flags_path = Path(os.getcwd()) / f'data/vae_model/{config["experiment_dir_bin"]}/flags.rar'
    return torch.load(flags_path)


def print_beta_value():
    flags = get_model_flags()
    return str(flags.beta)


def print_class_dim():
    flags = get_model_flags()
    return str(flags.class_dim)


def print_lr():
    flags = get_model_flags()
    # print(flags.__dict__['initial_learning_rate'])
    return str(flags.initial_learning_rate)


def print_img_shape():
    flags = get_model_flags()
    return f'({flags.img_size}, {flags.img_size})'


def print_flag_attribute(which: str):
    flags = get_model_flags()
    if which == 'img_size':
        return f'({flags.img_size}, {flags.img_size})'
    return float_to_tex(flags.__dict__[which], max_len=3)


def make_cond_gen_fig(nbr_samples=3):
    import mimic
    from mimic.utils import utils
    from mimic.utils.experiment import MimicExperiment
    from mimic.utils.filehandling import set_paths
    log.info(f'Starting generating cond gen fig with nbr_samples={nbr_samples}')
    config = get_config()

    # set seed
    SEED = config['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # experiment_dir = config['experiment_dir_bin']
    experiment_dir = 'binary_labels-True_beta-0.01_weighted_sampler-False_class_dim-128_text_gen_lastlayer-softmax_2021_02_10_14_56_27_974859'
    experiment_path = Path(__file__).parent.parent / f'data/vae_model/{experiment_dir}'
    flags_path = experiment_path / 'flags.rar'
    FLAGS = torch.load(flags_path)
    FLAGS.save_figure = True
    FLAGS.dir_cond_gen = Path(__file__).parent.parent / 'data/cond_gen'
    # FLAGS.text_gen_lastlayer = 'softmax'

    FLAGS = set_paths(FLAGS)
    FLAGS.use_clf = False
    FLAGS.binary_labels = False
    state_dict_path = experiment_path / 'checkpoints/0149/mm_vae'

    mimic_experiment = MimicExperiment(flags=FLAGS)
    mimic_experiment.mm_vae.to(FLAGS.device)
    mimic_experiment.mm_vae.load_state_dict(state_dict=torch.load(state_dict_path))
    mimic_experiment.mm_vae.eval()

    mimic_experiment.modalities['text'].plot_img_size = torch.Size([1, 256, 128])

    samples = mimic_experiment.test_samples
    model = mimic_experiment.mm_vae
    mods = mimic_experiment.modalities
    subsets = mimic_experiment.subsets

    if not Path(mimic_experiment.flags.dir_cond_gen).exists():
        Path(mimic_experiment.flags.dir_cond_gen).mkdir()

    def create_cond_gen_plot(in_mods='Lateral_PA'):
        subset = subsets[in_mods]
        plot = {**{f'in_{mod}': [] for mod in mimic_experiment.modalities},
                **{f'out_{mod}': [] for mod in mimic_experiment.modalities}}

        for idx in range(nbr_samples):
            sample = samples[idx]

            i_batch = {
                mod.name: sample[mod.name].unsqueeze(0)
                for mod in subset
            }
            latents = model.inference(i_batch, num_samples=1)
            c_in = latents['subsets'][in_mods]
            c_rep = utils.reparameterize(mu=c_in[0], logvar=c_in[1])
            cond_mod_in = {'content': c_rep, 'style': {k: None for k in mimic_experiment.modalities}}
            cond_gen_samples = model.generate_from_latents(cond_mod_in)
            for mod_key, mod in mods.items():
                plot[f'in_{mod_key}'].append(mod.plot_data(mimic_experiment, sample[mod_key].squeeze(0)))
                plot[f'out_{mod_key}'].append(mod.plot_data(mimic_experiment, cond_gen_samples[mod_key].squeeze(0)))

        rec = torch.Tensor()

        # first concatenate all input images, then all the output images
        for which, modalities in {'in': mods, 'out': mods}.items():
            for mod in modalities:
                for idx in range(nbr_samples):
                    if mod == 'text':
                        img = plot[f'{which}_{mod}'][idx].cpu().unsqueeze(0)
                    else:

                        img = plot[f'{which}_{mod}'][idx].cpu()
                        # pad the non text modalities such that they fit in a wider rectangle.
                        m = nn.ZeroPad2d((64, 64, 0, 0))
                        img = m(img.squeeze()).unsqueeze(0).unsqueeze(0)
                    rec = torch.cat((rec, img), 0)

        out_path = Path(mimic_experiment.flags.dir_cond_gen) / f'{in_mods}{"_small" if nbr_samples < 5 else ""}.png'
        log.info(f'Saving image to {out_path}')

        _ = mimic.utils.plot.create_fig(out_path,
                                        img_data=rec,
                                        num_img_row=nbr_samples, save_figure=True)

    for in_mod in mimic_experiment.subsets:
        if in_mod:
            # for in_mod in ['Lateral_text']:
            create_cond_gen_plot(in_mod)


def print_rand_perf():
    config = get_config()
    dir_clf = Path(os.getcwd()) / f'data/clfs/{config["dir_clf"]}'

    clf_eval_results_path = dir_clf / f'clf_test_results_bin_label.json'

    with open(clf_eval_results_path, 'r') as json_file:
        clf_eval_results = json.load(json_file)
    return round(clf_eval_results['random_perf']['mean_AP_Finding'][0], 3)


if __name__ == '__main__':
    # print(print_flag_attribute('vocab_size'))
    print_rand_perf()
    # print(print_lr())
    # print(print_nofinding_counts())
