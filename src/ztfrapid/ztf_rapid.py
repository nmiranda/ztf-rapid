# -*- coding: utf-8 -*- 
import os
from copy import copy

import numpy as np
import pandas as pd
from astrorapid.ANTARES_object.constants import GOOD_PHOTFLAG, TRIGGER_PHOTFLAG
from astrorapid.neural_network_model import train_model
from astrorapid.prepare_training_set import PrepareTrainingSetArrays
from astrorapid.process_light_curves import InputLightCurve
from imblearn.over_sampling import RandomOverSampler

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

    return X_train, X_test, y_train, y_test

def augment_datasets(input_dirpath, random_state):

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

    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X_train_2d, labels_train)

    y_train_res = y_train[ros.sample_indices_]
    X_train_res = X_train[ros.sample_indices_]

    return X_train_res, X_test, y_train_res, y_test

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

    return model.predict(X_test)

def aggregate_runs():
    pass
    