make-dataset /home/nmiranda/workspace/ztf_rapid/data/raw/rcf_marshallc_sncosmo_200114_2018classupdate_addedcv.pkl /home/nmiranda/workspace/ztf_rapid/data/interim/test_over /home/nmiranda/workspace/ztf_rapid/data/interim/test_over/test_over.npz

augment-dataset /home/nmiranda/workspace/ztf_rapid/data/interim/test_over/test_over.npz /home/nmiranda/workspace/ztf_rapid/data/processed/test_over/test_over_01.npz --rand 42 --strategy oversample