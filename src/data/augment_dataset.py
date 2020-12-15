# -*- coding: utf-8 -*-
import click
from ztfrapid.ztf_rapid import augment_datasets
import numpy as np

@click.command()
@click.argument('input_dirpath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_dirpath, output_filepath):

    X_train_res, X_test, y_train_res, y_test = augment_datasets(input_dirpath)

    np.savez(
        output_filepath,
        X_train_res=X_train_res,
        X_test=X_test,
        y_train_res=y_train_res,
        y_test=y_test,
    )

    

if __name__ == '__main__':
    main()
