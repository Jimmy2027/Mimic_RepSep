# HK, 27.01.21
import json
import os
from pathlib import Path

import mimic
import torch
import torch.nn as nn
from PIL import ImageFont
from mimic.utils import utils
from mimic.utils.experiment import MimicExperiment
from mimic.utils.filehandling import set_paths

from utils import get_config_path

# config_path = '/home/hendrik/docsrc/mimic_repsep/configs/bartholin.json'
mimic_config_path = Path(os.getcwd()) / f'prepare/mimic_configs/{get_config_path()}.json'
with open(mimic_config_path, 'r') as json_file:
    config = json.load(json_file)

experiment_dir = '/mnt/data/hendrik/mimic_scratch/mimic/moe/non_factorized/class_dim-128_rec_weight_m3-0.8_feature_extractor_img-resnet_2021_01_21_02_39_40_309212'
flags_path = os.path.expanduser(os.path.join(experiment_dir, 'flags.rar'))
FLAGS = torch.load(flags_path)
FLAGS.save_figure = True
FLAGS.dir_cond_gen = '/home/hendrik/docsrc/mimic_repsep/data/cond_gen'
# temp
FLAGS.rec_weight_m1 = 0.33
FLAGS.rec_weight_m2 = 0.33
FLAGS.rec_weight_m3 = 0.33

FLAGS = set_paths(FLAGS)
FLAGS.use_clf = False
# alphabet_path = os.path.join(str(Path(os.getcwd())), 'alphabet.json')
# with open(alphabet_path) as alphabet_file:
#     alphabet = str(''.join(json.load(alphabet_file)))
state_dict_path = os.path.expanduser(os.path.join(experiment_dir, 'checkpoints/0149/mm_vae'))
print(os.path.exists(state_dict_path))
mimic_experiment = MimicExperiment(flags=FLAGS)
mimic_experiment.mm_vae.to(FLAGS.device)
mimic_experiment.mm_vae.load_state_dict(state_dict=torch.load(state_dict_path))
mimic_experiment.mm_vae.eval()

# mimic_experiment.modalities['text'].font = ImageFont.truetype(str(Path(__file__).parent / 'FreeSerif.ttf'), 20)
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

    plot_out = mimic.utils.plot.create_fig(os.path.join(mimic_experiment.flags.dir_cond_gen, f'{in_mods}.png'),
                                           img_data=rec,
                                           num_img_row=5, save_figure=True)


for in_mod in ['Lateral_PA_text', 'Lateral_PA']:
    create_cond_gen_plot(in_mod)
