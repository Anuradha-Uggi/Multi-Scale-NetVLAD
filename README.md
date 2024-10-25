# Multi-Scale NetVLAD
This contains codes for the reproducibility of the [Multi-Scale-NetVLAD](https://ieeexplore.ieee.org/document/10605600) work. The training codes of MS-NetVLAD on the GSV-Cities dataset can be accessed from the gsv-cities directory.  
# Summary of Multi-Scale (MS) NetVLAD
This work proposes a new technique to address the Visual Place Recognition (VPR) problem. MS-NetVLAD essentially aggregates the multi-scale features across multiple intermediate layers of the backbone model, exploiting the complete hierarchy of the backbone features. The simple and efficient MS-NetVLAD improves upon popular VPR models such as NetVLAD, Patch-NetVLAD, etc., by a considerable margin.       


## Repo details
The training implementation of the model on the GSV-Cities is inspired by [MixVPR implementation](https://github.com/amaralibey/MixVPR).
Current codes are for training the model on GSV-Cities. The details of training on other datasets (Pittsburgh 30k and MSLS) will be uploaded soon. 

## Data
Training data: [GSV-Cities](https://github.com/amaralibey/gsv-cities?tab=readme-ov-file), 
Test data: copy of the Pittsburgh 250k (available [here](https://github.com/Relja/netvlad/issues/42))

## Cluster
Make appropriate changes to the directory paths in the code snippets. 
The below command generates clusters to initialize the model. 
```python
python main.py --mode=cluster
```
## Train
This is to train the model. 
```python
python main.py --mode=train 
```

