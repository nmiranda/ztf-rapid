# -*- coding: utf-8 -*-
import os

import click
import numpy as np
from ztfrapid.ztf_rapid import train


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('output_dirpath', type=click.Path())
@click.option('--epochs', type=int, default=25)
def main(input_filepath, output_filepath, output_dirpath, epochs):

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    files = np.load(input_filepath)
    X_train = files['X_train']
    X_test = files['X_test']
    y_train = files['y_train']
    y_test = files['y_test']

    model = train(X_train, X_test, y_train, y_test, output_dirpath, epochs)

    model.save(output_filepath)

if __name__ == '__main__':
    main()
