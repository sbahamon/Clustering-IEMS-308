# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:48:42 2016

@author: Steffany
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

data = np.loadtxt("clean_providers.csv", delimiter = ",")

K = range(1,10)
meandistortions = []
for k in K:
    km = KMeans(n_clusters = k)
    km.fit(data)
    meandistortions.append(sum(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
