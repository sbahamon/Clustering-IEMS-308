# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:53:10 2016

@author: Steffany
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = np.loadtxt("clean_providers.csv", delimiter = ",")

#cluster
km = KMeans(n_clusters = 6).fit(data)
kmlabels = km.labels_
kmcentroids = km.cluster_centers_

silhouette_avg = silhouette_score(data, kmlabels, sample_size = 1000)
print("The average silhouette_score is :", silhouette_avg)
print(kmcentroids)
