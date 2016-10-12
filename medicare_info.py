# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:56:50 2016

@author: sbv092
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks


#import and clean data
data = np.loadtxt("clean_providers.csv", delimiter = ",")
patients = data[:, 1]
payment = data[:, 2]

#histogram of patients and payments
plt.figure(1)
plt.subplot(211)
plt.title('Patients Serviced')
plt.plot(patients)

plt.subplot(212)
plt.title('Payments Recieved')
plt.plot(payment)

#correlation heatmap
R = np.corrcoef(data, rowvar = 0)
pcolor(R)
colorbar()
yticks(np.arange(0.5,2.5),range(0,2))
xticks(np.arange(0.5,2.5),range(0,2))
show()
