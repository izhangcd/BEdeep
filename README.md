### System requirements

The code were tesed on Linux and Mac OS systems. The required software/packages are:

- python==3.10.8
- matplotlib==3.7.0
- logging==0.5.1.2
- pkbar==0.5
- torch==1.13.1
- numpy==1.23.5
- pandas==1.4.3
- Bio==1.78
- sklearn==1.1.1

It is worth noting that when the computing environment(e.g, the version of tensorflow or biopython) changes, the prediction results might change slightly, but the main conclusion won't be affected.

### Installation Guide

```shell
conda create -n pytorch python=3.10.8 ipykernel matplotlib=3.7.0 logging=0.5.1.2 pkbar=0.5 torch=1.13.1 numpy=1.23.5 pandas=1.4.3 Bio=1.78 sklearn=1.1.1
ipython kernel install --user --name crispr --display-name "Python3(pytorch)"
```

Installation time depends on your own network environment.

### Files description

[ABEdeepoff.ipynb](https://github.com/izhangcd/DeepHF/blob/master/ABEdeepoff.ipynb) provides the code for training model in your own computing environment for ABE.

[CBEdeepoff.ipynb](https://github.com/izhangcd/DeepHF/blob/master/CBEdeepoff.ipynb) provides the code for training model in your own computing environment for CBE.

[data/ABEdeepoff.txt](https://github.com/izhangcd/DeepHF/blob/master/data/ABEdeepoff.txt) experimental edit efficiency data for ABE. It can be used to train the model.

[data/CBEdeepoff.txt](https://github.com/izhangcd/DeepHF/blob/master/data/CBEdeepoff.txt) experimental edit efficiency data for CBE. It can be used to train the model.

[data/ABE_Off_endo.txt](https://github.com/izhangcd/DeepHF/blob/master/data/ABE_Off_endo.txt) experimental edit efficiency data from independent publication dataset. It can be used to test the model.

[data/CBE_Off_endo.txt](https://github.com/izhangcd/DeepHF/blob/master/data/CBE_Off_endo.txt) experimental edit efficiency data from independent publication dataset. It can be used to test the model.
