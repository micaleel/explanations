"""
This module walks through a given directory, find valid configuration files,
and produces explanations for them.

Example:
    Assuming the a directory etc/ containing many configuration files, the
    command below will dig through all folders in the directory, and produce
    explanations using each configuration file it finds.

        $ python runner.py --source etc/

"""
import logging
import subprocess
from argparse import ArgumentParser

from config import Config
from utils import timeit, walk_path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


@timeit
def main(source_dir: str):
    for path in walk_path(source_dir, file_extension='yml'):
        config = Config.from_file(path)
        logger.info('Processing configuration file at {}...'.format(path))
        command = 'python explain.py --config {config_file} --build_profiles > {log_file}'
        command = command.format(config_file=path, log_file=path.replace('.yml', '.log'))
        subprocess.call(command, shell=True)
    logger.info('Finished')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', dest='source_path', help='Directory containing configuration files')
    args = parser.parse_args()
    main(source_dir=args.source_path)
