#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimic

#python prepare/prepare_config.py || exit
#python prepare/make_cond_gen_fig.py || exit
python prepare/test_vae_gen.py || exit
python prepare/test_vae_lr.py || exit
python prepare/test_clfs_new.py || exit
conda deactivate

./cleanup.sh || exit
./compile.sh slides || exit
echo "produce workflow finished"

