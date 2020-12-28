# -*- coding: utf-8 -*-
import click
import numpy as np
from kerastuner.tuners import RandomSearch
from ztfrapid import ztf_rapid


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('report_dirpath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, report_dirpath, output_filepath):
    
    input = np.load(input_filepath)

    y_train = input['X_train']
    y_train = input['X_test']
    y_train = input['y_train'][:,:,1:]
    y_test = input['y_test'][:,:,1:]
    num_classes = len(files['class_names'])

    hypermodel = ztf_rapid.HyperRAPID(num_classes)

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=1,
        executions_per_trial=1,
        directory=report_dirpath,
        project_name='ztf_rapid')

    tuner.search(files['X_train'], y_train, epochs=1, validation_data=(files['X_test'], y_test))

    best_hp = tuner.get_best_hyperparameters()[0].values
    
    with open(output_filepath, 'w') as fp:
        json.dump(best_hp, fp)

if __name__ == '__main__':
    main()
