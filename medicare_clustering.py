# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:21:01 2016

@author: sbv092
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.loadtxt("clean_providers.csv", delimiter = ",")

#cluster
km = KMeans(n_clusters = 6).fit(data)
km.fit(data)

reduced_data = PCA(n_components = 2).fit_transform(data)
r_km = KMeans(n_clusters = 6).fit(reduced_data)
