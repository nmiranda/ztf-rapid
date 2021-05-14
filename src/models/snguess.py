# -*- coding: utf-8 -*-

import click
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split


# def objective(dtrain, params):

#     model = xgb.cv(params, dtrain)



@click.command()
@click.option('--learning-rate', type=float, default=0.3, help="learning rate to update step size at each boosting step (default: 0.3)")
@click.option('--colsample-bytree', type=float, default=1.0, help="subsample ratio of columns when constructing each tree (default: 1.0)")
@click.option('--subsample', type=float, default=1.0, help="subsample ratio of the training instances (default: 1.0)")
def main(learning_rate, colsample_bytree, subsample):

    mlflow.set_experiment("SNGuess")

    X_cols = [
        'cut_pp',
        'ndet',
        'mag_det',
        'mag_last',
        't_lc',
        'rb_med',
        'drb_med',
        'distnr_med',
        'magnr_med',
        'classtar_med',
        'sgscore1_med',
        'distpsnr1_med',
        'sgscore2_med',
        'distpsnr2_med',
        'neargaia_med',
        'maggaia_med',
        'bool_pure',
        't_predetect',
        'bool_peaked',
        'mag_peak',
        'bool_norise',
        'bool_rising',
        'bool_hasgaps',
        'slope_rise_g',
        'slope_rise_r',
        'col_det',
        'col_last',
        'col_peak',
        'slope_fall_g',
        'slope_fall_r',
    ]
    
    features = pd.read_csv('/home/nmiranda/workspace/ztf_rapid/data/raw/snguess_features.csv')
    features.drop(columns='Unnamed: 0', inplace=True)

    X = features[X_cols]
    y = features['rcf_sn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    with mlflow.start_run():

        params = {
            "objective": "binary:hinge",
            "learning_rate": learning_rate,
            "eval_metric": "error",
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        y_pred = model.predict(dtest)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metrics({"precision": prec, "f1_score": f1})


if __name__ == '__main__':
    main()
