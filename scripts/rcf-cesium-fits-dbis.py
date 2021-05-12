import pandas as pd
import numpy as np
from cesium.features import GENERAL_FEATS, CADENCE_FEATS, LOMB_SCARGLE_FEATS
from cesium.featurize import featurize_time_series
from astropy.table import Table

bands = ('p48g', 'p48r', 'p48i')

lc_data = pd.read_pickle('/home/miranda/ztf-rapid/data/interim/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv_nozeroes.pkl')

for band in bands:
    ztfid_list = list()
    times_list = list()
    values_list = list()
    errors_list = list()
    target_list = list()
    for ztfid, lc in lc_data.items():
        lc = lc[lc['band'] == band]
        if len(lc) < 2:
            continue
        ztfid_list.append(ztfid)
        times_list.append(lc['mjd'])
        values_list.append(lc['flux'])
        errors_list.append(lc['flux'])
        target_list.append(lc.meta['classification'])

    features = featurize_time_series(times=times_list, values=values_list, errors=errors_list, features_to_use=GENERAL_FEATS + CADENCE_FEATS + LOMB_SCARGLE_FEATS, scheduler='single-threaded')

    features.columns = features.columns.droplevel(1)
    features['ztfid'] = ztfid_list
    features = features.set_index('ztfid')
    features['target'] = target_list

    feats_table = Table.from_pandas(features)

    feats_table.write('/home/miranda/ztf-rapid/data/interim/rcf_cesium_features_{band}.fits', format='fits', overwrite=True)

