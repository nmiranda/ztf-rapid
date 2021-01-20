# -*- coding: utf-8 -*-
import click
import numpy as np
import pandas as pd
from ztfrapid import ztf_rapid


@click.command()
@click.argument('metadata_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(metadata_filepath, data_filepath, output_filepath):

    lc_metadata = pd.read_csv(metadata_filepath)

    lc_metadata['label'] = lc_metadata['true_target'].apply(lambda x: ztf_rapid.PLASTICC_CLASSMAP_DICT[x])
    lc_metadata = lc_metadata[lc_metadata['label'].notna()]

    lc_data = pd.read_csv(data_filepath)

    lc_data_metadata = pd.merge(lc_metadata, lc_data)
    dataset = ztf_rapid.plasticc_make_datasets(lc_data_metadata, savedir='/tmp')

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
    )

if __name__ == '__main__':
    main()
