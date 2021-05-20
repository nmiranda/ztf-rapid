# -*- coding: utf-8 -*-

import click
import mlflow
import mlflow.tracking
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, fmin, hp, rand
from mlflow.tracking.client import MlflowClient
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import GroupShuffleSplit
from ztfrapid.ztf_rapid import plot_confusion_matrix
import matplotlib.pyplot as plt

plt.rc('text.latex', preamble=r'\usepackage{underscore}')

_inf = np.finfo(np.float64).max

LC_FEATS = [
    'cut_pp',
    'ndet',
    'mag_det',
    'mag_last',
    't_lc',
    'rb_med',
    'drb_med',
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

ALERT_FEATS = [
    'distnr_med',
    'magnr_med',
    'classtar_med',
    'sgscore1_med',
    'distpsnr1_med',
    'sgscore2_med',
    'distpsnr2_med',
    'neargaia_med',
    'maggaia_med',
]

class CVCallback(xgb.callback.TrainingCallback):

    def __init__(self, eval_metric):
        super().__init__()
        self.eval_metric = eval_metric
    
    def after_iteration(self, model, epoch, evals_log):

        mlflow.log_metric('train_error', evals_log['train'][self.eval_metric][-1][0], epoch)
        mlflow.log_metric('train_error_std', evals_log['train'][self.eval_metric][-1][1], epoch)
        mlflow.log_metric('test_error', evals_log['test'][self.eval_metric][-1][0], epoch)
        mlflow.log_metric('test_error_std', evals_log['test'][self.eval_metric][-1][1], epoch)

        return False

    def after_training(self, model):
        # print(model.cvfolds)
        # mlflow.log_param('after_training', 0.0)
        # mlflow.xgboost.log_model(model, artifact_path='.')
        return model

def get_objective_fn(dtrain, experiment_id, params):

    def objective(params):

        cv_callback = CVCallback(eval_metric=params['eval_metric'])

        num_boost = params['n_estimators']
        params.pop('n_estimators')

        with mlflow.start_run(nested=True, experiment_id=experiment_id):
            xgb_cv = xgb.cv(
                params=params,
                dtrain=dtrain,
                # num_boost_round=num_boost,
                seed=42,
                callbacks=[cv_callback],
                nfold=3
                )

        # loss = np.mean(xgb_cv['test-error-mean'])
        loss = xgb_cv.loc[:,'test-' + str(params['eval_metric']) + '-mean'].iloc[-1]
        # loss_variance = np.var(xgb_cv['test-error-mean'], ddof=1)
        loss_variance = xgb_cv.loc[:,'test-' + str(params['eval_metric']) + '-std'].iloc[-1]
        return {'loss': loss, 'loss_variance': loss_variance, 'status': STATUS_OK}
    
    return objective

@click.command()
# @click.option('--learning-rate', type=float, default=0.3, help="learning rate to update step size at each boosting step (default: 0.3)")
# @click.option('--colsample-bytree', type=float, default=1.0, help="subsample ratio of columns when constructing each tree (default: 1.0)")
# @click.option('--subsample', type=float, default=1.0, help="subsample ratio of the training instances (default: 1.0)")
@click.option('--no-alert-feats', is_flag=True)
@click.option('--task', type=click.Choice(['filter', 'classes']), default='filter')
@click.option('--test-size', type=float, default=0.3)
def main(no_alert_feats, task, test_size):

    mlflow.set_experiment("SNGuess")
    # mlflow.xgboost.autolog()

    if no_alert_feats:
        X_cols = LC_FEATS
    else:
        X_cols = LC_FEATS + ALERT_FEATS
    
    features = pd.read_csv('/home/nmiranda/workspace/ztf_rapid/data/raw/snguess_features.csv')
    features.drop(columns='Unnamed: 0', inplace=True)

    X = features[X_cols]

    if task == 'filter':
        rcf_data = pd.read_pickle('/home/nmiranda/workspace/ztf_rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl')
        y = features['snname'].apply(lambda x: x in rcf_data)
        objective = "binary:hinge"
        eval_metric = "error"
        num_class = None
        average = 'binary'
        class_names = ['non-RCF', 'RCF']
        groups = features['snname']
    else:
        y = 1*features['rcf_sn'] + 2*features['rcf_agn'] + 3*features['rcf_cv']
        mask_class_only = y != 0
        X = X[mask_class_only]
        y = y[mask_class_only]
        y = y - 1
        objective = "multi:softmax"
        eval_metric = "merror"
        num_class = 3
        average = 'weighted'
        class_names = ['SN', 'galaxy', 'stellar']
        groups = features[mask_class_only]['snname']

    group_shuffle_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(group_shuffle_split.split(X, y, groups=groups))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    with mlflow.start_run() as run:
        mlflow.set_tag("task", task)
        mlflow.set_tag("no_alert_feats", no_alert_feats)
        experiment_id = run.info.experiment_id

        mlflow.log_param('total_num_det', len(X))
        mlflow.log_param('num_sources', len(features['snname'].unique()))
        mlflow.log_param('min_jd', features['jd_det'].min())
        mlflow.log_param('max_jd', features['jd_det'].max())
        mlflow.log_param('test_size', test_size)
        mlflow.log_param('target_counts', str(y.value_counts().to_dict()))
        
        params = {
            "objective": objective,
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
            'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            "learning_rate": hp.quniform('learning_rate', 0.01, 0.5, 0.01),
            "eval_metric": eval_metric,
            "colsample_bytree": hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            "subsample": hp.quniform('subsample', 0.1, 1, 0.01),
            "seed": 42,
            "num_class": num_class
        }

        best = fmin(
            fn=get_objective_fn(dtrain, experiment_id, params),
            space=params,
            # max_evals=50,
            max_evals=3,
            algo=rand.suggest,
            rstate=np.random.RandomState(seed=42),
        )

        # mlflow.set_tag("best_params", str(best))

        params.update(best)
        params.pop('n_estimators')

        mlflow.log_param('best_params', str(best))

        model = xgb.train(params, dtrain)
        y_pred = model.predict(dtest)
        prec = precision_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)

        mlflow.log_metrics({"precision": prec, "f1_score": f1})

        plot_confusion_matrix(y_test, y_pred, class_names=class_names, normalize=None).figure_.savefig('confusion_matrix.svg')
        mlflow.log_artifact('confusion_matrix.svg')

        precision_ndet = list()
        for ndet in range(3,50):
            mask = X_test['ndet'] == ndet
            X_test_ndet = X_test[mask]
            y_test_ndet = y_test[mask]
            dtest_ndet = xgb.DMatrix(X_test_ndet, label=y_test_ndet)
            y_pred_ndet = model.predict(dtest_ndet)
            precision_ndet.append(precision_score(y_test_ndet, y_pred_ndet, average=average))

        fig, ax = plt.subplots(figsize=(6.0/1.5, 4.8/1.5), dpi=200)
        ax.plot(range(3,50), precision_ndet)
        ax.set_ylim(bottom=0.0, top=1.1)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Number of detections")
        fig.tight_layout()
        fig.savefig('precision_ndet.svg')
        mlflow.log_artifact('precision_ndet.svg')

        precision_jd = list()
        days_since_first_det = X_test['t_lc'].apply(lambda x: int(x))
        for days in range(2,50):
            mask = days_since_first_det == days
            X_test_jd = X_test[mask]
            y_test_jd = y_test[mask]
            dtest_jd = xgb.DMatrix(X_test_jd, label=y_test_jd)
            y_pred_jd = model.predict(dtest_jd)
            precision_jd.append(precision_score(y_test_jd, y_pred_jd, average=average))

        fig, ax = plt.subplots(figsize=(6.0/1.5, 4.8/1.5), dpi=200)
        ax.plot(range(2,50), precision_jd)
        ax.set_ylim(bottom=0.0, top=1.1)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Days since first detection")
        fig.tight_layout()
        fig.savefig('precision_jd.svg')
        mlflow.log_artifact('precision_jd.svg')

        # client = MlflowClient()
        # runs = client.search_runs([experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))
        # best_val_train = _inf
        # best_val_valid = _inf
        # best_val_test = _inf
        # best_run = None
        # for r in runs:
        #     print(r.data.metrics)
        #     if r.data.metrics["val_error"] < best_val_valid:
        #         best_run = r
        #         best_val_train = r.data.metrics["train_error"]
        #         best_val_valid = r.data.metrics["val_error"]
        #         best_val_test = r.data.metrics["test_error"]
        # mlflow.set_tag("best_run", best_run.info.run_id)
        # mlflow.log_metrics(
        #     {
        #         "train_error": best_val_train,
        #         "val_error": best_val_valid,
        #         "test_error": best_val_test,
        #     }
        # )

        # params = {
        #     "objective": "binary:hinge",
        #     "learning_rate": learning_rate,
        #     "eval_metric": "error",
        #     "colsample_bytree": colsample_bytree,
        #     "subsample": subsample,
        #     "seed": 42,
        # }
        # model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

        # y_pred = model.predict(dtest)
        # prec = precision_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)

        # mlflow.log_metrics({"precision": prec, "f1_score": f1})


if __name__ == '__main__':
    main()
