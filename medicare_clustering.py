# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:21:01 2016

@author: sbv092
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = np.loadtxt("clean_providers.csv", delimiter = ",")

#cluster
km = KMeans(n_clusters = 6).fit(data)
kmlabels = km.labels_

silhouette_avg = silhouette_score(data, kmlabels, sample_size = 1000)
print("The average silhouette_score is :", silhouette_avg)
