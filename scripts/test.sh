make-dataset /home/nmiranda/workspace/ztf_rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/nmiranda/workspace/ztf_rapid/data/interim/test

augment-dataset /home/nmiranda/workspace/ztf_rapid/data/interim/test /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_01.npz --rand 42 --strategy undersample
augment-dataset /home/nmiranda/workspace/ztf_rapid/data/interim/test /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_02.npz --rand 42 --strategy undersample
augment-dataset /home/nmiranda/workspace/ztf_rapid/data/interim/test /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_03.npz --rand 42 --strategy undersample
augment-dataset /home/nmiranda/workspace/ztf_rapid/data/interim/test /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_04.npz --rand 42 --strategy undersample

train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_01.npz /home/nmiranda/workspace/ztf_rapid/models/test_under/test_under_01.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test
train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_02.npz /home/nmiranda/workspace/ztf_rapid/models/test_under/test_under_02.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test
train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_03.npz /home/nmiranda/workspace/ztf_rapid/models/test_under/test_under_03.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test
train-model /home/nmiranda/workspace/ztf_rapid/data/processed/test_under/test_under_04.npz /home/nmiranda/workspace/ztf_rapid/models/test_under/test_under_04.hdf5 /home/nmiranda/workspace/ztf_rapid/reports/figures/test