make-dataset /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test

augment-dataset /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/processed/test_over/test_over_01.npz --rand 42 --strategy undersample
augment-dataset /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/processed/test_over/test_over_02.npz --rand 43 --strategy undersample
augment-dataset /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/processed/test_over/test_over_03.npz --rand 44 --strategy undersample
augment-dataset /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/processed/test_over/test_over_04.npz --rand 45 --strategy undersample

train-model /home/miranda/ztf-rapid/data/processed/test_over/test_over_01.npz /home/miranda/ztf-rapid/models/test_over/test_over_01.hdf5 /home/miranda/ztf-rapid/reports/figures/test
train-model /home/miranda/ztf-rapid/data/processed/test_over/test_over_02.npz /home/miranda/ztf-rapid/models/test_over/test_over_02.hdf5 /home/miranda/ztf-rapid/reports/figures/test
train-model /home/miranda/ztf-rapid/data/processed/test_over/test_over_03.npz /home/miranda/ztf-rapid/models/test_over/test_over_03.hdf5 /home/miranda/ztf-rapid/reports/figures/test
train-model /home/miranda/ztf-rapid/data/processed/test_over/test_over_04.npz /home/miranda/ztf-rapid/models/test_over/test_over_04.hdf5 /home/miranda/ztf-rapid/reports/figures/test