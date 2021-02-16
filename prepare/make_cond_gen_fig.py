# HK, 27.01.21
import os
import random
from pathlib import Path

import mimic
import numpy as np
import torch
import torch.nn as nn
from mimic.utils import utils
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import set_paths

from logger.logger import log
from utils import get_config


def make_cond_gen_fig():
    config = get_config()

    # set seed
    SEED = config['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    experiment_dir = config['experiment_dir_bin']
    experiment_path = Path(os.getcwd()) / f'data/vae_model/{experiment_dir}'
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

    def create_cond_gen_plot(in_mods='Lateral_PA', nbr_samples=5):
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
        out_path = Path(mimic_experiment.flags.dir_cond_gen) / f'{in_mods}.png'
        log.info(f'Saving image to {out_path}')
        _ = mimic.utils.plot.create_fig(out_path,
                                        img_data=rec,
                                        num_img_row=5, save_figure=True)

    for in_mod in ['Lateral_PA_text', 'Lateral_text']:
        # for in_mod in ['Lateral_text']:
        create_cond_gen_plot(in_mod)


if __name__ == '__main__':
    make_cond_gen_fig()
