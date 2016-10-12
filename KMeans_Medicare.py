# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:28:11 2016

@author: Steffany
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#import and clean data
df = pd.read_csv('providers.csv')
df = df[df['Entity Type of the Provider'] == 'I']

df = df[['Gender of the Provider', 
       'Medicare Participation Indicator', 
       'Number of Services', 
       'Number of Medicare Beneficiaries',
       'Total Submitted Charge Amount', 
       'Total Medicare Allowed Amount',
       'Total Medicare Payment Amount',
       'Total Medicare Standardized Payment Amount', 
       'Number of Female Beneficiaries', 
       'Number of Male Beneficiaries']]
       
df['Gender of the Provider'] = df['Gender of the Provider'].map({'M': 1, 'F':0})
df['Medicare Participation Indicator'] = df['Medicare Participation Indicator'].map({'Y': 1, 'N':0})
df = df.fillna(0)
data = df.as_matrix()
del df

#cluster
km = KMeans(n_clusters = 4)
km.fit(data)

reduced_data = PCA(n_components = 2).fit_transform(data)
kmeans = KMeans(n_clusters = 4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()