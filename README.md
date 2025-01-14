# Multi-Scale (MS) NetVLAD
This repository provides the implementation of the [Multi-Scale-NetVLAD](https://ieeexplore.ieee.org/document/10605600) approach. The development of MS-NetVLAD follows [MixVPR](https://github.com/amaralibey/MixVPR), [NetVLAD](https://github.com/Nanne/pytorch-NetVlad?tab=readme-ov-file), and [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD) repos.

## Summary
This work proposes a new technique to address the Visual Place Recognition (VPR) problem. MS-NetVLAD essentially aggregates the multi-scale features across multiple intermediate layers of the backbone model, exploiting the complete hierarchy of the backbone features. The simple and efficient MS-NetVLAD improves upon popular VPR models such as NetVLAD, Patch-NetVLAD, etc., by a considerable margin.       

## Repo. details
The current code is for training the model on the GSV-Cities dataset. Details for training on other datasets, including Pittsburgh 30k and MSLS, will be uploaded soon.

## Data
Training data: [GSV-Cities](https://github.com/amaralibey/gsv-cities?tab=readme-ov-file), 
Test data: [Pittsburgh 250k](https://github.com/Relja/netvlad/issues/42)

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
## Bibtex
Please use the below BibTeX to cite if you use the code.
```
@ARTICLE{10605600,
  author={Uggi, Anuradha and Channappayya, Sumohana S.},
  journal={IEEE Signal Processing Letters}, 
  title={MS-NetVLAD: Multi-Scale NetVLAD for Visual Place Recognition}, 
  year={2024},
  volume={31},
  number={},
  pages={1855-1859},
  keywords={Visualization;Image recognition;Transforms;Contrastive learning;Benchmark testing;Feature extraction;Vectors;Image matching;visual place recognition;scale invariance;NetVLAD},
  doi={10.1109/LSP.2024.3425279}}
```
