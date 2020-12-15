make-dataset /home/nmiranda/workspace/ztf_rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/nmiranda/workspace/ztf_rapid/data/interim/test
augment-dataset /home/nmiranda/workspace/ztf_rapid/data/interim/test /home/nmiranda/workspace/ztf_rapid/data/processed/test.npz

train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test.npz /home/nmiranda/workspace/ztf_rapid/models/test_01.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test
train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test.npz /home/nmiranda/workspace/ztf_rapid/models/test_02.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test
train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test.npz /home/nmiranda/workspace/ztf_rapid/models/test_03.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test
train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test.npz /home/nmiranda/workspace/ztf_rapid/models/test_04.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test