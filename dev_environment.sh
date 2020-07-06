#!/bin/bash

conda create --name jax-gpu2 python=3.7 jupyter scipy matplotlib numpy tqdm joblib numba cudatoolkit=10.0 cupy dask nose tensorflow-gpu h5py conda-build pytest

conda install -c conda-forge jupyter_contrib_nbextensions papermill plotnine panel bokeh holoviews hvplot param datashader funcy dask-jobqueue tiledb-py watermark opt_einsum

pip install cityhash

conda install -c conda-forge distributed=2.3 cloudpickle fastparquet fsspec murmurhash partd pyarrow toolz xxhash

pip install nbimporter more_itertools sparse

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
