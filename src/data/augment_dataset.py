# -*- coding: utf-8 -*-
import os

import click
import numpy as np
from ztfrapid.ztf_rapid import augment_datasets


@click.command()
@click.argument('input_dirpath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--rand', default=42, help='Random state integer.')
def main(input_dirpath, output_filepath, rand):

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    X_train_res, X_test, y_train_res, y_test = augment_datasets(input_dirpath, random_state=rand)

    np.savez(
        output_filepath,
        X_train_res=X_train_res,
        X_test=X_test,
        y_train_res=y_train_res,
        y_test=y_test,
    )

    

if __name__ == '__main__':
    main()
