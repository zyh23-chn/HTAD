# Heterogeneous Topological-Aware Debiasing
Implementation of ECAI 2025 paper "Exploring Topological Bias in Heterogeneous Graph Neural Networks".

## Data preparation
Download datasets provided by [1].

[1] Q. Lv, M. Ding, Q. Liu, Y. Chen, W. Feng, S. He, C. Zhou, J. Jiang, Y. Dong, and J. Tang. Are we really making much progress? revisiting, benchmarking and refining heterogeneous graph neural networks. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, pages 1150–1160, 2021.

## Investigate topological bias in HGNNs
```
cd src/scripts
bash run_deg.py
```

## Run the proposed method on three datasets
ACM dataset
```
cd src/scripts
bash run_acm.sh
```
IMDB dataset
```
cd src/scripts
bash run_imdb.sh
```
DBLP dataset
```
cd src/scripts
bash run_dblp.sh
```
