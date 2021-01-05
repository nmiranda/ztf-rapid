# -*- coding: utf-8 -*-
import os

import click
import numpy as np
from ztfrapid.ztf_rapid import augment_datasets


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--rand', default=42, help='Random state integer.')
@click.option('--strategy', type=click.Choice(['undersample', 'oversample']), required=True)
def main(input_filepath, output_filepath, rand, strategy):

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # X_train_res, X_test, y_train_res, y_test, objids_test, class_names = augment_datasets(
    #     input_filepath, 
    #     random_state=rand, 
    #     strategy=strategy
    #     )

    files = np.load(input_filepath, allow_pickle=True)

    X_train_res, y_train_res = augment_datasets(
        files['X_train'],
        files['y_train'],
        files['labels_train'],
        random_state=rand,
        strategy=strategy,
    )

    np.savez(
        output_filepath,
        X_train=X_train_res,
        X_test=files['X_test'],
        y_train=y_train_res,
        y_test=files['y_test'],
        objids_test=files['objids_test'],
        class_names=files['class_names'],
    )

if __name__ == '__main__':
    main()
