# -*- coding: utf-8 -*-
import click
from tensorflow.keras.models import load_model
from ztfrapid.ztf_rapid import predict

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(model_filepath, input_filepath, output_filepath):

    model = load_model(model_filepath)
    files = np.load(input_filepath)
    X_test = files['X_test']

    y_pred = predict(model, X_test)

    np.save(output_filepath, y_pred)

if __name__ == '__main__':
    main()