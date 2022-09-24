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
