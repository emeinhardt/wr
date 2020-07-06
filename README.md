# wr

This repository contains code for creating a model of the posterior confusability of words given (1) a language model and (2) diphone gating data that can be used to define a noise model.
See [Meinhardt et al (2020)](https://www.aclweb.org/anthology/2020.acl-main.180.pdf) for more on the context of application and exposition of the mathematical model.

## Architecture Overview

The processing pipeline of this repository involves
 - a directory setup containing output data from a few preprocessing repositories
 - a few key Python modules in the repository
 - a processing pipeline of Jupyter notebooks, mostly run through a driver notebook (courtesy of [Papermill](https://papermill.readthedocs.io/en/latest/)).
   - The code here was developed with [literate programming](https://www.wikiwand.com/en/Literate_programming) in mind: code, documentation, and tests are in-line, and use of `papermill` to create instances of a notebook applied to particular input parameters creates an environment where you can understand how a computation proceeded (or an error arose) and drop in to inspect it. 

The main driver notebook has much more detail.

## Setup

**A brief note on hardware:** As noted in more granular detail elsewhere, much of the code in this repository is developed on research servers with 60-200 GB of RAM, a GPU, and ≈1TB available for (completely uncompressed) generated data.
(In many cases, peak RAM usage is a function of naive and safe parallelization via `joblib`; asking `joblib` to use fewer jobs/cores will often result in much lower peak RAM usage.)
All code has been developed on/for a *nix-like OS.
As a general rule, processing steps earlier in the pipeline (especially focused on the noise model) are more likely to be feasible or require less adaptation for leaner hardware.

1. Create an appropriate conda environment (following shell commands in 'dev_environment.sh' or using the conda environment in 'wr_env.yml').
2. Clone, configure, and run the main notebooks of each of the following repositories to create preprocessed data for code in this repository:
  - <https://github.com/emeinhardt/wmc2014-ipa>
  - <https://github.com/emeinhardt/switchboard-lm>
  - <https://github.com/emeinhardt/buckeye-lm>
  - <https://github.com/emeinhardt/fisher-lm>
  - <https://github.com/emeinhardt/fisher-lm-srilm>
3. Open the 'Processing Driver Notebook' and follow further instructions there.

As documented in the repositories, the preprocessing notebooks all assume you have rights and access to the relevant underlying datasets.
