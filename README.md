# Multi-Scale NetVLAD
This contains codes for the reproducibility of the [Multi-Scale-NetVLAD](https://ieeexplore.ieee.org/document/10605600) work. The training codes of MS-NetVLAD on the GSV-Cities dataset can be accessed from the gsv-cities directory.  

## Data
Training data: [GSV-Cities](https://github.com/amaralibey/gsv-cities?tab=readme-ov-file)
Test data: copy of the Pittsburgh 250k (available [here](https://github.com/Relja/netvlad/issues/42))

## Usage
Use the command below to generate clusters to initialize the model. 
```python
python main.py --mode=cluster 

Use the command below to train the model
```python 
main.py --mode=train 

