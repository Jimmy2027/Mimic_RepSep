import argparse
import shutil
import sys
from glob import glob
from pathlib import Path

filename2zotkey = {
    "midl_slides": "EQU7CI79",
    "midl-shortpaper": "9UVPMZAM"
}

zot_storage_path = Path('~/Zotero/storage').expanduser()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='file to upload and display')
    flags = parser.parse_args()

    file_path = Path(flags.file)
    file_name = file_path.stem

    dest_dir = zot_storage_path / filename2zotkey[file_name]
    dest_filename = Path(glob(str(dest_dir / "*"))[0]).name

    shutil.copy(file_path, dest_dir / dest_filename)


if __name__ == '__main__':
    sys.exit(main())
