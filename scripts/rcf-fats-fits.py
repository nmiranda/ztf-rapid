import pandas as pd
import numpy as np
from FATS.Feature import FeatureSpace
from astropy.table import Table

bands = ('p48g', 'p48r', 'p48i')

lc_data = pd.read_pickle('/home/miranda/ztf-rapid/data/interim/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv_nozeroes.pkl')

for band in bands:
    ztfid_list = list()
    target_list = list()
    results_list = list()
    for ztfid, lc in lc_data.items():
        lc = lc[lc['band'] == band]
        if len(lc) < 2:
            continue
        this_fluxes = lc['flux']
        this_times = lc['mjd']
        this_errors = lc['fluxerr']
        feature_space = FeatureSpace(Data='all', featureList=None, excludeList=['interp1d'])
        feature_space = feature_space.calculateFeature(np.array([this_fluxes, this_times, this_errors]))
        this_result = feature_space.result()
        # results_list.append(feature_space.result())
        print(Table(this_result))
        exit(0)

    