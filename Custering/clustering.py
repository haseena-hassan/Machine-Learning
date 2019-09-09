# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 09:38:15 2019

@author: Jesna
"""
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.75, random_state=0)
#n_samples : int, optional (default=100)
#centers : int or array of shape [n_centers, n_features].The number of centers to generate, or the fixed center locations.
#cluster_std : float or sequence of floats, optional (default=1.0). The standard deviation of the clusters.
#random_state : int, RandomState instance or None (default)
#Determines random number generation for dataset creation. 
plt.scatter(X[:, 0], X[:, 1], s=50);
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
#cmap= viridis, plasma,inferno,magma
#
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);