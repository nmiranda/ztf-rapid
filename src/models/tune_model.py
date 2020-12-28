# -*- coding: utf-8 -*-
import json

import click
import numpy as np
from kerastuner.tuners import RandomSearch
from ztfrapid import ztf_rapid


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('report_dirpath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, report_dirpath, output_filepath):
    
    input_data = np.load(input_filepath)

    X_train = input_data['X_train']
    X_test = input_data['X_test']
    y_train = input_data['y_train'][:,:,1:]
    y_test = input_data['y_test'][:,:,1:]
    num_classes = len(input_data['class_names'])

    hypermodel = ztf_rapid.HyperRAPID(num_classes)

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=1,
        executions_per_trial=1,
        directory=report_dirpath,
        project_name='ztf_rapid')

    tuner.search(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

    best_hp = tuner.get_best_hyperparameters()[0].values
    
    with open(output_filepath, 'w') as fp:
        json.dump(best_hp, fp)

if __name__ == '__main__':
    main()
