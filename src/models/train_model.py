# -*- coding: utf-8 -*-
import click
import numpy as np
from ztfrapid.ztf_rapid import train


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('output_dirpath', type=click.Path())
@click.option('--rand', default=42, help='Random state integer.')
def main(input_filepath, output_filepath, output_dirpath, rand):
    
    files = np.load(input_filepath)
    X_train_res = files['X_train_res']
    X_test = files['X_test']
    y_train_res = files['y_train_res']
    y_test = files['y_test']

    model = train(X_train_res, X_test, y_train_res, y_test, output_dirpath)

    model.save(output_filepath)

if __name__ == '__main__':
    main()
