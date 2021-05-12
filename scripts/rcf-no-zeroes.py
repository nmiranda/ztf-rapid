import pandas as pd

lc_data = pd.read_pickle('/home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl')

lc_data_no_zeroes = {ztfid: new_lc for ztfid, new_lc in ((ztfid, lc[lc['flux'] > 0.0]) for ztfid, lc in lc_data.items()) if len(new_lc) > 0}

pd.to_pickle(lc_data_no_zeroes, '/home/miranda/ztf-rapid/data/interim/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv_nozeroes.pkl')