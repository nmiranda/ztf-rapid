make-dataset-plasticc /home/miranda/ztf-rapid/data/raw/plasticc_train_metadata.csv.gz /home/miranda/ztf-rapid/data/raw/plasticc_train_lightcurves.csv.gz /home/miranda/ztf-rapid/data/interim/test_plasticc_over/test_plasticc_over.npz

augment-dataset /home/miranda/ztf-rapid/data/interim/test_plasticc_over/test_plasticc_over.npz /home/miranda/ztf-rapid/data/processed/test_plasticc_over/test_plasticc_over_01.npz --rand 42 --strategy oversample

train-model /home/miranda/ztf-rapid/data/processed/test_plasticc_over/test_plasticc_over_01.npz /home/miranda/ztf-rapid/models/test_plasticc_over/test_plasticc_over_01.hdf5 /home/miranda/ztf-rapid/reports/figures/test_plasticc_over