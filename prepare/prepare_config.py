# HK, 15.02.21
import os
import tempfile
from pathlib import Path

import numpy as np

from logger.logger import log
from utils import write_to_config


def set_seed():
    rand_seed = np.random.randint(0, 1000)
    write_to_config({'seed': rand_seed})


def init_data():
    """
    If data folder doesn't exists, downloads it and extracts it.
    """
    data_path = Path(os.getcwd()) / 'data'
    if not data_path.exists():
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_name = 'e7f9b8ef73f5.zip'
            wget_command = f'wget https://ppb.hendrikklug.xyz/{zip_name} -P {tmpdirname}/'
            log.info(f'Executing wget command: {wget_command}')
            os.system(wget_command)
            unzip_command = f'unzip {tmpdirname}/{zip_name} -d {data_path.parent}/'
            log.info(f'Unzipping data folder with: {unzip_command}')
            os.system(unzip_command)
    assert data_path.exists()


if __name__ == '__main__':
    set_seed()
    init_data()
