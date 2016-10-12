"""
Created on Mon Oct  3 14:28:11 2016
@author: Steffany
"""

import pandas as pd
import numpy as np

#import and clean data
df = pd.read_csv('providers.csv')
df = df[df['Entity Type of the Provider'] == 'I']
df = df[df['Medicare Participation Indicator'] == 'Y']

df = df[['Gender of the Provider', 
       'Number of Services', 
       'Total Medicare Standardized Payment Amount']]
       
df['Gender of the Provider'] = df['Gender of the Provider'].map({'M': 1, 'F':0})
df = df.fillna(0)
df.describe()

data = df.as_matrix()

np.savetxt("clean_providers.csv", data, delimiter = ",")
