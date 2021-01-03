# -*- coding: utf-8 -*-

import click
import pandas as pd
import numpy as np
from ztfrapid import ztf_rapid

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('dataset_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.option('--epc', default=50, type=int, help='Number of elements per class.')
def main(input_filepath, dataset_filepath, output_filepath, epc):

    lc_data_orig = pd.read_pickle(input_filepath)
    datasets = np.load(dataset_filepath, allow_pickle=True)

    lc_data = ztf_rapid.noisify_dataset(datasets['objids_train'], datasets['labels_train'], lc_data_orig, elem_per_class=epc)
    
    datasets_aug = ztf_rapid.make_datasets(lc_data, '/tmp/', split_data=False)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    np.savez(
        output_filepath,
        X_train=datasets_aug['X'],
        X_test=datasets['X_test'],
        y_train=datasets_aug['y'],
        y_test=datasets['y_test'],
        objids_test=datasets['objids_test'],
        class_names=datasets['class_names'],
    )