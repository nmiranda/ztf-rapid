# -*- coding: utf-8 -*- 
import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astrorapid.ANTARES_object.constants import GOOD_PHOTFLAG, TRIGGER_PHOTFLAG
from astrorapid.neural_network_model import train_model
from astrorapid.prepare_training_set import PrepareTrainingSetArrays
from astrorapid.process_light_curves import InputLightCurve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics

CLASS_MAP = {
    1: 'SN Ia',
    2: 'SN II',
    3: 'AGN',
    4: 'CV'
}
CLASS_MAP_REV = {
    'SN Ia': 1,
    'SN II': 2,
    'AGN': 3,
    'CV': 4
}
BANDS = {'p48g': "g", 'p48r': "r", 'p48i': "i"}
BANDS_VEC = np.vectorize(lambda x: BANDS[x])

def to_classnumber(lc):
    class_name = lc.meta['classification']
    try:
        return CLASS_MAP_REV[class_name]
    except KeyError as e:
        pass
    return None

def to_astrorapid(this_lc):

    this_lc = copy(this_lc)
    
    mask = np.logical_or.reduce((this_lc['band'] == 'p48r', this_lc['band'] == 'p48g', this_lc['band'] == 'p48i'))
    this_lc = this_lc[mask]
    mask = this_lc['flux'] != 0.0
    this_lc = this_lc[mask]
    
    if len(this_lc) <= 2:
        return None

    this_lc['band'] = BANDS_VEC(this_lc['band'])

    this_lc['photflag'] = [TRIGGER_PHOTFLAG] + (len(this_lc)-1) * [GOOD_PHOTFLAG]
    
    class_number = to_classnumber(this_lc)
    
    return InputLightCurve(
        mjd=this_lc['mjd'], 
        flux=this_lc['flux'], 
        fluxerr=this_lc['fluxerr'], 
        passband=this_lc['band'], 
        photflag=this_lc['photflag'], 
        ra=this_lc.meta['ra'], 
        dec=this_lc.meta['dec'],
        objid=this_lc.meta['ztfname'],
        redshift=None,
        mwebv=this_lc.meta['mwebv'],
        known_redshift=False,
        training_set_parameters={
            'class_number': class_number,
            'peakmjd': this_lc[np.argmax(this_lc['flux'])]['mjd']
        }
    )

def make_datasets(filepath, savedir):
    """
    docstring
    """
    
    lc_data = pd.read_pickle(filepath)

    def get_data(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo, calculate_t0):

        lightcurves = {str(class_num) + '_' + str(key): to_astrorapid(val) for key, val in lc_data.items() if to_classnumber(val) == class_num}
        valid_lightcurves = {key: val.preprocess_light_curve() for key, val in lightcurves.items() if val is not None}
        
        return valid_lightcurves

    preparearrays = PrepareTrainingSetArrays(
        reread_data=False,
        class_name_map=CLASS_MAP,
        training_set_dir=os.path.join(savedir, 'training'),
        data_dir=os.path.join(savedir, 'data'),
        save_dir=os.path.join(savedir, 'save'),
        get_data_func=get_data,
        contextual_info=(),
        nobs=150,
        mintime=0,
        maxtime=150,
        timestep=1.0,
        passbands=('g', 'r', 'i'),
    )

    X_train, X_test, y_train, y_test, labels_train, \
    labels_test, class_names, class_weights, sample_weights, \
    timesX_train, timesX_test, orig_lc_train, orig_lc_test, \
    objids_train, objids_test = preparearrays.prepare_training_set_arrays(class_nums=tuple(CLASS_MAP.keys()))

    return X_train, X_test, y_train, y_test, objids_test, class_names

def augment_datasets(input_dirpath, random_state, strategy='oversample'):

    preparearrays = PrepareTrainingSetArrays(
        reread_data=False,
        class_name_map=CLASS_MAP, 
        training_set_dir=os.path.join(input_dirpath, 'training'),
        data_dir=os.path.join(input_dirpath, 'data'),
        save_dir=os.path.join(input_dirpath, 'save'),
        # get_data_func=get_data,
        get_data_func=None,
        contextual_info=(),
        nobs=150,
        mintime=0,
        maxtime=150,
        timestep=1.0,
        passbands=('g', 'r', 'i'),
    )

    X_train, X_test, y_train, y_test, labels_train, \
    labels_test, class_names, class_weights, sample_weights, \
    timesX_train, timesX_test, orig_lc_train, orig_lc_test, \
    objids_train, objids_test = preparearrays.prepare_training_set_arrays(class_nums=tuple(CLASS_MAP.keys()))

    X_train_2d = X_train.transpose(0,2,1).reshape(-1,X_train.shape[2]*X_train.shape[1])

    if strategy == 'oversample':
        ros = RandomOverSampler(random_state=random_state)
    elif strategy == 'undersample':
        ros = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError('Strategy non valid.')
    X_res, y_res = ros.fit_resample(X_train_2d, labels_train)

    y_train_res = y_train[ros.sample_indices_]
    X_train_res = X_train[ros.sample_indices_]

    return X_train_res, X_test, y_train_res, y_test, objids_test, class_names

def train(X_train_res, X_test, y_train_res, y_test, output_dirpath):

    try:
        os.mkdir(output_dirpath)
    except FileExistsError as e:
        print(e)
    model = train_model(
        X_train_res,
        X_test,
        y_train_res,
        y_test,
        fig_dir=output_dirpath,
        # epochs=25,
        epochs=2,
        retrain=True
        # retrain=False
    )

    return model

def predict(model, X_test):

    #return model.predict(X_test)
    return model(X_test, training=False).numpy()

def count_results_for_class(results, class_):
    return np.sum(results.to_numpy()[:,1:] == class_, axis=1)

def runs_result_dataframe(y_test, y_pred_list, objids_test):
    
    y_pred_1d_list = [np.argmax(y_pred[:,-1,:], axis=1) for y_pred in y_pred_list]
    res_index = ["model_%s" % i for i in range(1,len(y_pred_1d_list)+1)]
    res_vals = pd.DataFrame({k: v for k, v in zip(res_index, y_pred_1d_list)})
    res_head = pd.DataFrame({
        'objid': objids_test,
        'class': np.argmax(y_test[:,0,:], axis=1),
    })
    res = pd.concat((res_head, res_vals), axis=1)
    res = res.set_index('objid')

    return res

def result_class_distribution(results, class_names):
    
    res_dist = pd.DataFrame({
            class_: count_results_for_class(results, idx+1) / len(class_names) \
                for idx, class_ in enumerate(class_names)
            }, 
        index=results.index)
    
    return res_dist
    
def true_pred_ensemble(y_test, y_pred_list, objids_test, class_names, cutoff=0.75):

    res = runs_result_dataframe(y_test, y_pred_list, objids_test)
    res_dist = result_class_distribution(res, class_names)

    mask_cut_fraction = np.any(res_dist.to_numpy() >= cutoff, axis=1)
    pred_cut = res_dist.iloc[mask_cut_fraction].to_numpy().argmax(axis=1) + 1
    test_cut = res.iloc[mask_cut_fraction].loc[:,'class'].to_numpy()

    return test_cut, pred_cut

def plot_confusion_matrix(y_true, y_pred, class_names):

    cm = metrics.confusion_matrix(
        y_true, 
        y_pred,
        normalize='true'
    )
    cmd = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    return cmd.plot(
        cmap=plt.cm.Blues, 
    )
