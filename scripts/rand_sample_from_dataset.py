# HK, 23.02.21
import os
import textwrap
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt

from prepare.utils import get_config, set_paths

config = get_config()
experiment_dir = config['experiment_dir']
experiment_path = Path(os.getcwd()) / f'data/vae_model/{experiment_dir}'
flags_path = experiment_path / 'flags.rar'
flags = torch.load(flags_path)
flags = set_paths(flags, config)

split = 'eval'
dir_dataset = os.path.join(flags.dir_data, f'files_small_{flags.img_size}')
fn_img_pa = os.path.join(dir_dataset, split + '_pa.pt')
fn_img_lat = os.path.join(dir_dataset, split + '_lat.pt')
fn_findings = os.path.join(dir_dataset, split + '_findings.csv')

imgs_pa = torch.load(fn_img_pa)
imgs_lat = torch.load(fn_img_lat)
report_findings = pd.read_csv(fn_findings)['findings']

img_pa = imgs_pa[0]
img_lat = imgs_lat[0]
text = report_findings[0]

lines = textwrap.wrap(text, width=40)

ax1 = plt.subplot(212)
# ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
# ax1.imshow(np.zeros((128,128)))
ax1.axis([0, 10, 0, 10])
ax1.text(2.5, 3.5, '\n'.join(lines), fontsize=12)
ax1.axis('off')
ax1.set_title('Text report',fontweight="bold")

ax2 = plt.subplot(221)
ax2.imshow(img_pa)
ax2.set_title('Lateral view',fontweight="bold")
ax2.axis('off')

ax3 = plt.subplot(222)
# ax3.margins(x=0, y=-0.25)   # Values in (-0.5, 0.0) zooms in to center
ax3.imshow(img_lat)
ax3.set_title('Frontal view',fontweight="bold")
ax3.axis('off')

plt.tight_layout(h_pad=3)
# plt.show()