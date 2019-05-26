#!/bin/bash

conda create --name anvil python=3.6 csvkit jupyter scipy matplotlib numpy tqdm joblib numba cudatoolkit=9.2 pyculib 
conda activate anvil
conda install -c conda-forge jupyter_contrib_nbextensions papermill plotnine panel bokeh holoviews hvplot param datashader funcy dask-jobqueue
pip install more_itertools
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

