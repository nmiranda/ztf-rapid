make-dataset /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test_noisify /home/miranda/ztf-rapid/data/interim/test_noisify/test_noisify.npz --nocv

noisify /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test_noisify/test_noisify.npz /home/miranda/ztf-rapid/data/processed/test_noisify_augtest/test_noisify_augtest.npz --epc 2000 --augtest

train-model /home/miranda/ztf-rapid/data/processed/test_noisify_augtest/test_noisify_augtest.npz /home/miranda/ztf-rapid/models/test_noisify_augtest/test_noisify_augtest.hdf5 /home/miranda/ztf-rapid/reports/figures/test_noisify_augtest