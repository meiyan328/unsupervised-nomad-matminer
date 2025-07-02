# Clustering Analysis of Compound Datasets

This repository contains code and visualizations for a research project focused on **unsupervised clustering of chemical compounds** using two well-known algorithms: **HDBSCAN** and **DPC (Density Peak Clustering)**. The analysis is performed on two major materials science datasets: **NOMAD** and **Matminer**.

## Project Overview

We systematically evaluate the clustering behavior of HDBSCAN and DPC on compound datasets by comparing their ability to identify meaningful groupings and outliers.

### Methods

- **Algorithms used**:
  - HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
  - DPC (Density Peak Clustering)
  
- **Datasets**:
  - [NOMAD](https://nomad-lab.eu)
  - [Matminer](https://hackingmaterials.lbl.gov/matminer/)

- **Evaluation metrics**:
  - Normalized Mutual Information (NMI)
  - Adjusted Mutual Information (AMI)
  - Adjusted Rand Index (ARI)
  - Jaccard Index

- **Dimensionality reduction for visualization**:
  - PCA (Principal Component Analysis)
  - t-SNE (t-distributed Stochastic Neighbor Embedding)

- **Visual tools**:
  - Heatmaps
  - Colormaps
  - Scatter plots

##  Goals

- Compare the **cluster structures** produced by HDBSCAN and DPC.
- Assess the **robustness against noise and outliers**.
- Investigate the **distribution patterns** in reduced-dimensional space.
