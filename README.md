# Clustering-IEMS-308
Hi!

These are Python files I used in Fall of 2016 to complete a clustering homework assignment based off Medicare Provider Data from 2014.
I clustered on gender, number of patients seen, and total payment given to the physician to see if there was evidence of systematic payment discrimination. Turns out there's none.


Table of Contents:
medicare_data_cleaning.py - reads in the csv format, cleans it to find relevant features using pandas, converts it to a numpy array, saves                             it to a csv.
medicare_info.py -          finds statistical metrics and histograms for every column in the data set
medicare_elbow_plot.py -    creates an elbow plot to find optimal number of clusters.
medicare_clustering.py -    clusters the data based off above attributes, printing the centroids and the silhouette score of the                                       clustering
