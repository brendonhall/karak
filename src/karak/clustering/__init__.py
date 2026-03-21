"""Clustering pipeline: PCA dimensionality reduction + HDBSCAN phase detection.

Submodules
----------
hdbscan_cluster
    Core HDBSCAN clustering.
noise_assign
    k-NN noise reassignment.
pca
    PCA dimensionality reduction.
tiling
    Tiled progressive HDBSCAN with phase registry unification.
"""
