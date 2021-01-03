make-dataset /home/nmiranda/workspace/ztf_rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/nmiranda/workspace/ztf_rapid/data/interim/test_noisify /home/nmiranda/workspace/ztf_rapid/data/interim/test_noisify/test_noisify.npz --nocv

noisify /home/nmiranda/workspace/ztf_rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/nmiranda/workspace/ztf_rapid/data/interim/test_noisify/test_noisify.npz /home/nmiranda/workspace/ztf_rapid/data/processed/test_noisify/test_noisify.npz --epc 100

train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test_noisify/test_noisify.npz /home/nmiranda/workspace/ztf_rapid/models/test_noisify/test_noisify.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test_noisify