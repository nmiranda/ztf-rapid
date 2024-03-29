# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from ztfrapid.ztf_rapid import make_datasets


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_dirpath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.option('--nocv', is_flag=True)
def main(input_filepath, output_dirpath, output_filepath, nocv):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    if nocv:
        class_nums = (1,2,3)
    else:
        class_nums = None

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    project_dir = Path(__file__).resolve().parents[2]

    # savedir = '/home/nmiranda/workspace/ztf_rapid/data/interim/test'

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    lc_data = pd.read_pickle(input_filepath)

    dataset = make_datasets(lc_data, output_dirpath, class_nums=class_nums)

    np.savez(
        output_filepath,
        X_train=dataset['X_train'],
        X_test=dataset['X_test'],
        y_train=dataset['y_train'],
        y_test=dataset['y_test'],
        objids_test=dataset['objids_test'],
        objids_train=dataset['objids_train'],
        orig_lc_test=np.array(dataset['orig_lc_test'], dtype=object),
        orig_lc_train=np.array(dataset['orig_lc_train'], dtype=object),
        timesX_test=dataset['timesX_test'],
        class_names=dataset['class_names'],
        labels_train=dataset['labels_train'],
        labels_test=dataset['labels_test'],
    )


if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
