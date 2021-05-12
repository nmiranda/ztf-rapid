import pandas as pd
import numpy as np
from FATS.Feature import FeatureSpace
from astropy.table import Table
import dask

dask.distributed.client = Client(threads_per_worker=4, n_workers=1)

def get_features(feature_space, fluxes, times, errors):
    this_fs = feature_space.calculateFeature(np.array([fluxes, times, errors]))
    return this_fs.result(method='dict')

bands = ('p48g', 'p48r', 'p48i')
# exclude_list = ['interp1d', 'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80', 'PercentDifferenceFluxPercentile']
exclude_list = ['interp1d']

lc_data = pd.read_pickle('/home/miranda/ztf-rapid/data/interim/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv_nozeroes.pkl')

for band in bands:

    ztfid_list = list()
    target_list = list()
    results_list = list()

    for ztfid, lc in lc_data.items():
        lc = lc[lc['band'] == band]
        if len(lc) < 2:
            continue
        ztfid_list.append(ztfid)
        target_list.append(lc.meta['classification'])
        this_fluxes = lc['flux']
        this_times = lc['mjd']
        this_errors = lc['fluxerr']
        feature_space = FeatureSpace(Data=['magnitude', 'time', 'error'], featureList=None, excludeList=exclude_list)
        # feature_space = feature_space.calculateFeature(np.array([this_fluxes, this_times, this_errors]))
        # this_result = feature_space.result(method='dict')
        this_result = dask.delayed(get_features)(this_fluxes, this_times, this_errors)
        results_list.append(this_result)

    results_list = dask.compute(results_list)

    features = pd.DataFrame(results_list)
    features['ztfid'] = ztfid_list
    features['target'] = np.array(target_list, dtype='U')

    feats_table = Table.from_pandas(features)

    feats_table.write(f'/home/miranda/ztf-rapid/data/interim/rcf_fats_features_{band}.fits', format='fits', overwrite=True)
    