from setuptools import find_packages, setup

setup(
    name='ztfrapid',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    version='0.1.0',
    description='A short description of the project.',
    author='nmiranda',
    license='',
    entry_points={
        'console_scripts': [
            'make-dataset=data.make_dataset:main',
            'augment-dataset=data.augment_dataset:main',
            'train-model=models.train_model:main',
            'tune-model=models.tune_model:main',
            'noisify=data.noisify:main',
            'make-dataset-plasticc=data.make_plasticc_dataset:main',
            ],
    }
)
