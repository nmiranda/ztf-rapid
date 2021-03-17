# -*- coding: utf-8 -*-
import os

import click
import numpy as np
import pandas as pd
from ztfrapid import ztf_rapid
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('dataset_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.option('--epc', default=50, type=int, help='Number of elements per class.')
@click.option('--augtest', is_flag=True)
def main(input_filepath, dataset_filepath, output_filepath, epc, augtest):

    lc_data_orig = pd.read_pickle(input_filepath)
    datasets = np.load(dataset_filepath, allow_pickle=True)

    lc_data = ztf_rapid.noisify_dataset(datasets['objids_train'], datasets['labels_train'], lc_data_orig, elem_per_class=epc)

    datasets_aug = ztf_rapid.make_datasets(lc_data, '/tmp/', split_data=False)

    X_train = datasets_aug['X']
    y_train = datasets_aug['y']

    if augtest:
        lc_data_test = ztf_rapid.noisify_dataset(datasets['objids_test'], datasets['labels_test'], lc_data_orig, elem_per_class=epc)
        datasets_train_aug = ztf_rapid.make_datasets(lc_data_test, '/tmp/', split_data=False)

        X_test = datasets_train_aug['X']
        y_test = datasets_train_aug['y']

    else:

        X_test = datasets['X_test']
        y_test = datasets['y_test']

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    np.savez(
        output_filepath,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        objids_test=datasets['objids_test'],
        class_names=datasets['class_names'],
    )
