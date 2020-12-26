make-dataset /home/miranda/ztf-rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/interim/test/test.npz

time parallel -j 10 augment-dataset /home/miranda/ztf-rapid/data/interim/test /home/miranda/ztf-rapid/data/processed/test_over/test_over_{}.npz --rand {} --strategy oversample ::: $(seq -w 1 100)

time parallel -j 5 train-model /home/miranda/ztf-rapid/data/processed/test_over/test_over_{}.npz /home/miranda/ztf-rapid/models/test_over/test_over_{}.hdf5 /home/miranda/ztf-rapid/reports/figures/test_over ::: $(seq -w 1 100)