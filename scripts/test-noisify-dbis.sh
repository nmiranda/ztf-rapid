make-dataset /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test_noisify /home/miranda/ztf-rapid/data/interim/test_noisify/test_noisify.npz --nocv

noisify /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test_noisify/test_noisify.npz /home/miranda/ztf-rapid/data/processed/test_noisify/test_noisify.npz --epc 2000

train-model /home/miranda/ztf-rapid/data/processed/test_noisify/test_noisify.npz /home/miranda/ztf-rapid/models/test_noisify/test_noisify.hdf5 /home/miranda/ztf-rapid/reports/figures/test_noisify