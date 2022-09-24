### System requirements
The code were tesed on Linux and Mac OS systems.

The required software/packages are:
* python>=3.6.5
* numpy=1.19.1
* Pytorch=1.11.0
* scikit-learn
* biopython=1.79
* pandas
* pkbar


### Help
- training.py Off-target models training for ABEmax and AncBE4max

  -b {ABE,CBE} Set base editor model training
- bedeepoff.py Off-target models for ABEmax and AncBE4max

  -b {ABE,CBE} Set base editor model

  -i INPUT_FILE Set input tsv file
### Example Call
- Off-target ABE/CBE model training:
```bash
python training.py -b ABE
python training.py -b CBE
```
- Off-target ABE prediction for Cas-OFFinder output:
```bash
python bedeepoff.py -b ABE -i ./cas-offinder/Cas-Offinder_Output_Example.txt
```
- Off-target ABE prediction for CRISPRitz output:
```bash
python bedeepoff.py -b ABE -i ./cas-offinder/CRISPRitz_Output_Example.txt
```
### Files description
* [bedeepoff.py](https://github.com/izhangcd/BEdeep/blob/main/bedeepoff.py) contains the contains the core code of prediction module for the website [www.DeepHF.com](http://www.deephf.com).
* [training.py](https://github.com/izhangcd/BEdeep/blob/main/training.py) contains the code for ABEdeepoff and CBEdeepoff model training.
* [ABEdeepoff.txt](https://github.com/izhangcd/BEdeep/blob/main/Data/ABEdeepoff.txt) contains the raw data for ABEdeepoff model training.
* [CBEdeepoff.txt](https://github.com/izhangcd/BEdeep/blob/main/Data/CBEdeepoff.txt) contains the raw data for CBEdeepoff model training.
* [ABEdeepoff.pt](https://github.com/izhangcd/BEdeep/blob/main/Models/ABEdeepoff.pt) the final model file of ABEdeepoff used in the DeepHF wibsite.
* [CBEdeepoff.pt](https://github.com/izhangcd/BEdeep/blob/main/Models/CBEdeepoff.pt) the final model file of CBEdeepoff used in the DeepHF wibsite.
