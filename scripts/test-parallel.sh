make-dataset /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/interim/test/test.npz

augment-dataset /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/processed/test_over/test_over_01.npz --rand 42 --strategy undersample

parallel -j 3 train-model /home/miranda/ztf-rapid/data/processed/test_over/test_over_01.npz /home/miranda/ztf-rapid/models/test_over/test_over_{}.hdf5 /home/miranda/ztf-rapid/reports/figures/test_over ::: 01 02 03