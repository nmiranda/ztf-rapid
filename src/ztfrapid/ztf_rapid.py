# -*- coding: utf-8 -*- 
import os
import random
import string
from collections import defaultdict
from copy import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits as pf
from astropy.table import Table
from astrorapid.ANTARES_object.constants import GOOD_PHOTFLAG, TRIGGER_PHOTFLAG
from astrorapid.neural_network_model import train_model
from astrorapid.prepare_training_set import PrepareTrainingSetArrays
from astrorapid.process_light_curves import InputLightCurve
from FATS.Feature import FeatureSpace
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from kerastuner import HyperModel
from scipy.stats import binned_statistic
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

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
PLASTICC_CLASSMAP = {
    90: "SN Ia",
    42: "SN II",
    88: "AGN",
    64: "KN",
}
PLASTICC_CLASSMAP_NUM = {
    "SN Ia": 1,
    "SN II": 2,
    "AGN": 3,
    "KN": 4,
}
PLASTICC_CLASSMAP_NUM_REV = {
    1: "SN Ia",
    2: "SN II",
    3: "AGN",
    4: "KN",
}
BANDMAP = {
    0: 'u',
    1: 'g',
    2: 'r',
    3: 'i',
    4: 'z',
    5: 'y',
}
BANDS = {'p48g': "g", 'p48r': "r", 'p48i': "i"}
BANDS_VEC = np.vectorize(lambda x: BANDS[x])
COLPB_ZTF = {'p48g': 'tab:green', 'p48r': 'tab:red', 'p48i': 'tab:blue'}
COLPB = {'g': 'tab:green', 'r': 'tab:red', 'i': 'tab:blue'}
COLORS = ['grey', 'tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:purple', 'tab:brown', '#aaffc3', 'tab:olive',
          'tab:cyan', '#FF1493', 'navy', 'tab:pink', 'lightcoral', '#228B22', '#aa6e28', '#FFA07A']
PLASTICC_BANDS_VEC = np.vectorize(lambda x: BANDMAP[x])
PLASTICC_CLASSMAP_DICT = defaultdict(lambda: None)
PLASTICC_CLASSMAP_DICT.update(PLASTICC_CLASSMAP)

class HyperRAPID(HyperModel):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        
        model = keras.Sequential()
        model.add(layers.Masking(mask_value=0.))
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(layers.LSTM(hp.Choice('units', [25,50,100]), return_sequences=True, dropout=hp.Choice('dropout', [0.0,0.1,0.2,0.3])))
        model.add(layers.TimeDistributed(layers.Dense(self.num_classes, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

def target_to_categorical(target):
    return np.argmax(target[:,0,:], axis=1)

def pred_to_categorical(pred):
    return np.argmax(pred[:,-1,:], axis=1)

def filter_pps(lightcurve):
    
    mask = np.logical_or.reduce((lightcurve['band'] == 'p48r', lightcurve['band'] == 'p48g', lightcurve['band'] == 'p48i'))
    lightcurve = lightcurve[mask]
    mask = lightcurve['flux'] != 0.0
    lightcurve = lightcurve[mask]

    return lightcurve

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

def make_datasets(lc_data, savedir, split_data=True, class_nums=None):
    """
    docstring
    """
    if not class_nums:
        class_nums = tuple(CLASS_MAP.keys())

    def get_data(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo, calculate_t0):

        lightcurves = {str(class_num) + '_' + str(key): to_astrorapid(val) for key, val in lc_data.items() if to_classnumber(val) == class_num}

        valid_lightcurves = dict()
        for key, val in lightcurves.items():
            if val is not None:
                try:
                    this_preproc_lc = val.preprocess_light_curve()
                except ValueError as e:
                    print(e)
                    continue
                valid_lightcurves[key] = this_preproc_lc
        
        return valid_lightcurves

    preparearrays = PrepareTrainingSetArrays(
        reread_data=True,
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

    if not split_data:
        X, y, labels, class_names, class_weights, sample_weights, timesX, orig_lc, \
            objids = preparearrays.prepare_training_set_arrays(class_nums=class_nums, split_data=False)

        return {
            'X': X,
            'y': y,
            'labels': labels,
            'class_names': class_names,
            'class_weights': class_weights,
            'sample_weights': sample_weights,
            'timesX': timesX,
            'orig_lc': orig_lc,
            'objids': objids,
        }

    X_train, X_test, y_train, y_test, labels_train, \
    labels_test, class_names, class_weights, sample_weights, \
    timesX_train, timesX_test, orig_lc_train, orig_lc_test, \
    objids_train, objids_test = preparearrays.prepare_training_set_arrays(class_nums=class_nums)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'orig_lc_test': orig_lc_test,
        'objids_test': objids_test,
        'timesX_test': timesX_test,
        'class_names': class_names,
        'objids_train': objids_train,
        'orig_lc_train': orig_lc_train,
        'labels_train': labels_train,
        'labels_test': labels_test,
    }

def augment_datasets(X, y, labels, random_state, strategy='oversample'):

    X_2d = X.transpose(0,2,1).reshape(-1,X.shape[2]*X.shape[1])

    if strategy == 'oversample':
        ros = RandomOverSampler(random_state=random_state)
    elif strategy == 'undersample':
        ros = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError('Strategy non valid.')
    ros.fit_resample(X_2d, labels)

    y_res = y[ros.sample_indices_]
    X_res = X[ros.sample_indices_]

    return X_res, y_res

def scale_3d(array):

    orig_shape = array.shape
    scaler = MinMaxScaler()
    array_normalized = scaler.fit_transform(array.reshape((orig_shape[0],-1))).reshape(orig_shape)
    return array_normalized

def train(X_train, X_test, y_train, y_test, output_dirpath, epochs=25):

    try:
        os.mkdir(output_dirpath)
    except FileExistsError as e:
        print(e)
    model = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        fig_dir=output_dirpath,
        epochs=epochs,
        retrain=True,
        # retrain=False,
        # workers=1,
        # use_multiprocessing=True,
    )

    return model

def predict(model, X_test):

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
            class_: count_results_for_class(results, idx+1) \
                for idx, class_ in enumerate(class_names)
            }, 
        index=results.index)
    res_dist = res_dist.div(res_dist.sum(axis=1), axis=0)
    
    return res_dist
    
def true_pred_ensemble(y_test, y_pred_list, objids_test, class_names, cutoff=0.75):

    res = runs_result_dataframe(y_test, y_pred_list, objids_test)
    res_dist = result_class_distribution(res, class_names)

    mask_cut_fraction = np.any(res_dist.to_numpy() >= cutoff, axis=1)
    pred_cut = res_dist.iloc[mask_cut_fraction].to_numpy().argmax(axis=1) + 1
    test_cut = res.iloc[mask_cut_fraction].loc[:,'class'].to_numpy()

    return test_cut, pred_cut

def plot_confusion_matrix(y_true, y_pred, class_names, normalize='true'):

    cm = metrics.confusion_matrix(
        y_true, 
        y_pred,
        normalize=normalize
    )
    cmd = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(dpi=160)
    return cmd.plot(
        cmap=plt.cm.Blues,
        ax=ax,
    )

def get_mag(lc):
    lc = lc[lc['flux'] > 0.0]
    return np.array(-2.5 * np.log10(lc['flux'])+ lc['zp'])

def get_mag_err(lc):
    lc = lc[lc['flux'] > 0.0]
    return np.array(2.5 / np.log(10) * lc['fluxerr'] / lc['flux'])

def ztf_noisify(lightcurve, new_z=None, k_exp_scale=1.0, alpha=None):

    this_lc = copy(lightcurve)
    
    this_lc = this_lc[this_lc['flux'] > 0.0]
    
    if len(this_lc) == 0:
        return Table()
    
    this_lc['maglim'] = 20.8
    this_lc['maglim'][this_lc['band']=='p48g'] = 20.5
    this_lc['maglim'][this_lc['band']=='p48i'] = 20.2
    
    mag = -2.5 * np.log10(this_lc['flux'])+ this_lc['zp']
    magerr = 2.5 / np.log(10) * this_lc['fluxerr'] / this_lc['flux']
    maglim = this_lc['maglim']
    truez = this_lc.meta['z']

    if alpha:
        new_z = truez * alpha

    delta_m = cosmo.distmod(new_z)-cosmo.distmod(truez)
    
    k_exp = np.log(18+1) / (maglim-17)
    k_exp *= k_exp_scale
    
    df_f_oldexpected = 2 + np.exp(k_exp*(mag-17)) -1
    df_f_newexpected = 2 + np.exp(k_exp*(mag+delta_m.value-17)) -1
    noise_scaling = df_f_newexpected / df_f_oldexpected
    
    flux_true = 10**((25-mag)/2.5)
    df_f_true = np.maximum(0.02, (magerr * np.log(10) / 2.5))
    flux_new = 10**((25-mag-delta_m.value)/2.5)
    df_f_new = df_f_true * noise_scaling
    
    newfluxnoise = flux_new * (np.sqrt(df_f_new**2-df_f_true**2))
    
    flux_obs = flux_new + np.random.normal(scale=newfluxnoise)
    flux_err = newfluxnoise
    
    new_lc = Table()
    new_lc['mjd'] = this_lc['mjd']
    new_lc['band'] = this_lc['band']
    new_lc['flux'] = flux_obs
    new_lc['fluxerr'] = flux_err
    new_lc['zp'] = this_lc['zp']
    new_lc['zpsys'] = this_lc['zpsys']
    new_lc.meta = this_lc.meta
    new_lc.meta['z'] = new_z
    
    return new_lc

def plot_joint_real_noisified(lc_data, k_exp_scale, num_subset=100000):

    all_mag = np.concatenate([get_mag(lc) for lc in lc_data.values()])
    all_mag_err = np.concatenate([get_mag_err(lc) for lc in lc_data.values()])
    real_data = pd.DataFrame({'mag': all_mag, 'magErr': all_mag_err, 'type': 'Real'})

    all_sim_lc = [ztf_noisify(lc, lc.meta['z'] * 2.0, k_exp_scale) for lc in lc_data.values() if lc.meta['z'] is not None]
    sim_mag = np.concatenate([get_mag(lc) for lc in all_sim_lc if len(lc) > 0])
    sim_mag_err = np.concatenate([get_mag_err(lc) for lc in all_sim_lc if len(lc) > 0])
    sim_data = pd.DataFrame({'mag': sim_mag, 'magErr': sim_mag_err, 'type': 'Simulated'})

    sim_data_subset = sim_data[(sim_data['magErr'] > 0.0) & (sim_data['magErr'] < 0.5)]
    real_data_subset = real_data[(real_data['magErr'] > 0.0) & (real_data['magErr'] < 0.5)]
    sel_num = int(num_subset / 2)
    all_data_subset = pd.concat([real_data_subset[:sel_num], sim_data_subset[:sel_num]])

    range_min = all_data_subset['mag'].min()
    range_max = all_data_subset['mag'].max()

    real_bin_means, real_bin_edges, real_binnumber = binned_statistic(
        real_data_subset['mag'], 
        real_data_subset['magErr'],
        statistic='mean',
        range=(range_min, range_max),
    )
    real_bin_stds, _, _ = binned_statistic(
        real_data_subset['mag'], 
        real_data_subset['magErr'],
        statistic='std',
        range=(range_min, range_max),
    )
    real_bin_width = (real_bin_edges[1] - real_bin_edges[0])
    real_bin_centers = real_bin_edges[1:] - real_bin_width/2

    sim_bin_means, sim_bin_edges, sim_binnumber = binned_statistic(
        sim_data_subset['mag'], 
        sim_data_subset['magErr'],
        statistic='mean',
        range=(range_min, range_max),
    )
    sim_bin_stds, _, _ = binned_statistic(
        sim_data_subset['mag'], 
        sim_data_subset['magErr'],
        statistic='std',
        range=(range_min, range_max),
    )
    sim_bin_width = (sim_bin_edges[1] - sim_bin_edges[0])
    sim_bin_centers = sim_bin_edges[1:] - sim_bin_width/2
    sim_bin_centers = sim_bin_centers + 0.08

    g = sns.jointplot(
        data=all_data_subset, 
        x='mag', 
        y='magErr', 
        hue='type', 
        kind='kde',
    #     kind='scatter',
        fill=True,
    #     ylim=(-0.05, 0.5),
    #     xlim=(16, 24),
        alpha=0.6,
        legend='False',
    )

    g.ax_joint.errorbar(real_bin_centers[:8], real_bin_means[:8], yerr=real_bin_stds[:8], capsize=3.0, fmt='o')
    g.ax_joint.errorbar(sim_bin_centers[:8], sim_bin_means[:8], yerr=sim_bin_stds[:8], capsize=3.0, fmt='o')

def random_string(length=10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def noisify_id(objid, lc_data):

    objid = objid.split('_')[1]
    this_lc = lc_data[objid]
    try:
        noisified_lc = ztf_noisify(this_lc, alpha=2.0)
    except TypeError as e:
        return None
    noisified_lc.meta['ztfname'] = objid + '_' + random_string()
    
    return noisified_lc

def noisify_dataset(objids_train, labels_train, lc_data_orig, elem_per_class=50):

    objid_class = pd.DataFrame({'ztfid': objids_train, 'class_': labels_train})

    value_counts = objid_class['class_'].value_counts()

    augmented_list = list()
    for class_ in value_counts.index:
    # for class_ in ['AGN', 'SN II', 'SN Ia']:

        class_ids = objid_class.loc[objid_class['class_'] == class_,'ztfid']
        to_augment_ids = np.random.choice(class_ids, size=elem_per_class, replace=True)

        augmented_list += list(map(partial(noisify_id, lc_data=lc_data_orig), to_augment_ids))
    
    return {lc.meta['ztfname']: lc for lc in augmented_list if lc}

def get_pred_label_peak(X_test, y_pred):

    argmax_time = X_test.max(axis=-1).argmax(axis=-1)
    pred_label = y_pred[np.arange(y_pred.shape[0]),argmax_time].argmax(axis=-1)

    return pred_label

def get_pred_label(y_pred, time_index=-1):

    return y_pred[:,time_index,:].argmax(axis=-1)

def get_y_true(y_test):
    
    return y_test[:,0,:].argmax(axis=-1)

def plot_lightcurve_scores(lightcurve, timesX, X, y, objid, true_label, class_names):

    fig, (ax1, ax2) = plt.subplots(nrows=2, 
                                ncols=1, 
                                figsize=(6.0, 4.8),
                                dpi=200,
                                sharex=True)

    argmax = timesX.argmax() + 1
    for pbidx, pb in enumerate(BANDS.values()):
        if pb not in set(lightcurve['passband']):
            continue
        pbmask = lightcurve['passband'] == pb
        ax1.errorbar(lightcurve[pbmask]['time'], 
                    lightcurve[pbmask]['flux'], 
                    yerr=lightcurve[pbmask]['fluxErr'], 
                    fmt='o', 
                    label=pb, 
                    lw=2, 
                    markersize=5, 
                    alpha=0.8, 
                    c=COLPB[pb],
                    capsize=2.0,
                    )
        ax1.plot(timesX[:argmax], X[:, pbidx][:argmax], lw=3, c=COLPB[pb])

    for classnum, classname in enumerate(class_names):
        classnum = classnum+1
        ax2.plot(timesX[:argmax], y[:, classnum][:argmax], '-', label=classname,
                color=COLORS[classnum], linewidth=3)

    ax1.set_ylim(bottom=0.0)
    ax1.set_ylabel('Flux', fontsize='large')
    ax1.legend(loc='best', fontsize='medium')

    ax2.set_ylim((0.0, 1.0))
    ax2.set_ylabel('Class probability', fontsize='large')
    ax2.set_xlabel('Time from first detection (days)', fontsize='large')
    ax2.legend(loc='best', fontsize='medium')

    fig.suptitle("ID: %s, True class: %s" % (objid.split('_')[1], true_label),
                fontsize='medium')

def plot_processed_lightcurve(lightcurve, timesX=None, X=None):

    fig = plt.figure(figsize=(3.2, 2.4), dpi=200)

    for pbidx, pb in enumerate(BANDS.values()):
        if pb not in set(lightcurve['passband']):
            continue
        pbmask = lightcurve['passband'] == pb
        plt.errorbar(
            lightcurve[pbmask]['time'],
            lightcurve[pbmask]['flux'],
            yerr=lightcurve[pbmask]['fluxErr'],
            fmt='o',
            label=pb,
            lw=2,
            markersize=5,
            alpha=0.8,
            capsize=2.0,
            c=COLPB[pb]
        )

        if (timesX is not None) and (X is not None):
            argmax = timesX.argmax() + 1
            plt.plot(timesX[:argmax], X[:,pbidx][:argmax], c=COLPB[pb])

    plt.ylim(bottom=0.0)
    plt.ylabel('Flux')
    plt.xlabel('Time from first detection (days)')
    plt.legend(loc='upper right', title='Bands')
    fig.tight_layout()

    return fig


def plot_raw_lightcurve(lightcurve, ax=None):

    if ax is None:
        ax = plt.gca()

    for pbidx, pb in enumerate(BANDS.keys()):
        if pb not in set(lightcurve['band']):
            continue
        pbmask = lightcurve['band'] == pb
        ax.errorbar(
            lightcurve[pbmask]['mjd'],
            lightcurve[pbmask]['flux'],
            yerr=lightcurve[pbmask]['fluxerr'],
            fmt='o',
            label = pb,
            lw=2,
            markersize=5,
            alpha=0.8,
            capsize=2.0,
            c=COLPB_ZTF[pb]
        )

    ax.set_ylim(bottom=0.0)
    ax.set_ylabel('Flux')
    ax.set_xlabel('Time (MJD)')
    ax.legend(loc='upper right', title='Bands')
    ax.set_title("ZTFID: {}, Type: {}".format(lightcurve.meta['ztfname'], lightcurve.meta['classification']))

    return ax

def select_bright_sources(X, y, flux):

    mask = np.any(X.reshape(X.shape[0], -1) > flux, axis=-1)
    return X[mask], y[mask]

def plasticc_to_astrorapid(this_lc):

    mask = np.logical_or.reduce((this_lc['passband'] == 1, this_lc['passband'] == 2, this_lc['passband'] == 3))
    this_lc = this_lc[mask]
    mask = this_lc['flux'] > 0.0
    this_lc = this_lc[mask]

    if len(this_lc) <= 2:
        return None

    this_lc['band'] = PLASTICC_BANDS_VEC(this_lc['passband'])

    this_lc['photflag'] = [TRIGGER_PHOTFLAG] + (len(this_lc)-1) * [GOOD_PHOTFLAG]

    return InputLightCurve(
        mjd=this_lc['mjd'],
        flux=this_lc['flux'],
        fluxerr=this_lc['flux_err'],
        passband=this_lc['band'],
        photflag=this_lc['photflag'],
        ra=this_lc['ra'].iloc[0], 
        dec=this_lc['decl'].iloc[0],
        objid=this_lc['object_id'].iloc[0],
        redshift=None,
        mwebv=this_lc['mwebv'].iloc[0],
        known_redshift=False,
        training_set_parameters={
            'class_number': PLASTICC_CLASSMAP_NUM[this_lc['label'].iloc[0]],
            'peakmjd': this_lc.iloc[np.argmax(this_lc['flux'])]['mjd']
        }
    )

    return this_lc

def plasticc_to_classnumber(this_lc):

    return PLASTICC_CLASSMAP_NUM[this_lc['label'][0]]

def plasticc_make_datasets(lc_data, savedir, split_data=True, class_nums=None):

    if not class_nums:
        class_nums = tuple(CLASS_MAP.keys())

    def plasticc_get_data(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo, calculate_t0):

        lc_data[lc_data['label'] == PLASTICC_CLASSMAP_NUM_REV[plasticc_to_classnumber(lc_data)]]

        lightcurves = {str(class_num) + '_' + str(objid): plasticc_to_astrorapid(lc) for objid, lc in lc_data.groupby('object_id')}

        valid_lightcurves = dict()
        for key, val in lightcurves.items():
            if val is not None:
                try:
                    this_preproc_lc = val.preprocess_light_curve()
                except ValueError as e:
                    print(e)
                    continue
                valid_lightcurves[key] = this_preproc_lc
        
        return valid_lightcurves

    preparearrays = PrepareTrainingSetArrays(
        reread_data=True,
        class_name_map=PLASTICC_CLASSMAP_NUM_REV,
        training_set_dir=os.path.join(savedir, 'training'),
        data_dir=os.path.join(savedir, 'data'),
        save_dir=os.path.join(savedir, 'save'),
        get_data_func=plasticc_get_data,
        contextual_info=(),
        nobs=150,
        mintime=0,
        maxtime=150,
        timestep=1.0,
        passbands=('g', 'r', 'i'),
    )

    if not split_data:
        X, y, labels, class_names, class_weights, sample_weights, timesX, orig_lc, \
            objids = preparearrays.prepare_training_set_arrays(class_nums=class_nums, split_data=False)

        return {
            'X': X,
            'y': y,
            'labels': labels,
            'class_names': class_names,
            'class_weights': class_weights,
            'sample_weights': sample_weights,
            'timesX': timesX,
            'orig_lc': orig_lc,
            'objids': objids,
        }

    X_train, X_test, y_train, y_test, labels_train, \
    labels_test, class_names, class_weights, sample_weights, \
    timesX_train, timesX_test, orig_lc_train, orig_lc_test, \
    objids_train, objids_test = preparearrays.prepare_training_set_arrays(class_nums=class_nums)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'orig_lc_test': orig_lc_test,
        'objids_test': objids_test,
        'timesX_test': timesX_test,
        'class_names': class_names,
        'objids_train': objids_train,
        'orig_lc_train': orig_lc_train,
        'labels_train': labels_train,
    }

def generate_fits_features_band(input_filepath, output_filepath, band):

    lc_data = pd.read_pickle(input_filepath)

    feature_list = ['CAR_sigma', 'CAR_mean', 'Meanvariance', 'Mean', 'PercentAmplitude', 'Skew', 'AndersonDarling', 'Std', 'MedianAbsDev', 'Q31', 'Amplitude', 'PeriodLS']

    result_list = list()
    id_list = list()
    target_list = list()
    for this_id in lc_data.keys():
        this_lcs = lc_data[this_id]
        if len(this_lcs) < 1:
            continue
        this_lc = this_lcs[this_lcs['band'] == band]
        this_mag = get_mag(this_lc)
        this_mjd = np.array(this_lc[this_lc['flux'] > 0.0]['mjd'])
        this_mag_err = get_mag_err(this_lc)
        if len(this_mag) < 1:
            continue
        fields = [this_mag, this_mjd, this_mag_err]
        feature_space = FeatureSpace(featureList=feature_list)
        feature_space = feature_space.calculateFeature(np.array(fields))
        result_list.append(feature_space.result())
        id_list.append(this_id)
        target_list.append(this_lc.meta['classification'])

    features = pd.DataFrame(result_list, columns=feature_list)
    features['ztfid'] = id_list
    features['target'] = target_list

    table = features
    coldefs = [
        pf.Column(name='ztfid', format='12A', array=np.array(table['ztfid'])),
        pf.Column(name='CAR_mean', format='F', array=table['CAR_mean']),
        pf.Column(name='Meanvariance', format='F', array=table['Meanvariance']),
        pf.Column(name='Mean', format='F', array=table['Mean']),
        pf.Column(name='PercentAmplitude', format='F', array=table['PercentAmplitude']),
        pf.Column(name='Skew', format='F', array=table['Skew']),
        pf.Column(name='AndersonDarling', format='F', array=table['AndersonDarling']),
        pf.Column(name='Std', format='F', array=table['Std']),
        pf.Column(name='MedianAbsDev', format='F', array=table['MedianAbsDev']),
        pf.Column(name='Q31', format='F', array=table['Q31']),
        pf.Column(name='Amplitude', format='F', array=table['Amplitude']),
        pf.Column(name='PeriodLS', format='F', array=table['PeriodLS']),
        pf.Column(name='target', format='16A', array=table['target']),
    ]

    tbhdu = pf.BinTableHDU.from_columns(coldefs)
    tbhdu.writeto(output_filepath, checksum=True, overwrite=True)

def get_dataset_bands(lc_data):

    band_set = set()
    for ztf_id, lc in lc_data.items():
        band_set |= set(lc['band'])
    return band_set
