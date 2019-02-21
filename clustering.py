# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:22:07 2019

@author: Hong
"""
import pandas as pd
import numpy as np
import json
import re
import csv
from scipy.spatial.distance import hamming
from sklearn.cluster import KMeans 
"""
Variables
----------
auto: boolean
    detemine number of clusters automatically or not
    
n_cluster: int
    user-defined number of clusters

n_min : int
    minimum number of clusters, default value is 5

n_max: int 
    maximum number of clusters, default value is number of data/number of 
    point in each cluster
n_point: 
    how many point in each cluster, default value is 5
"""
auto=True
n_min = 5
n_max = 0
n_point = 5
n_cluster = 5


def vectorizer(data, list_roads):
    """
    Creat one-hot vector for each address in the data
    
    Parameters
    ----------
    data : (N,) list
        Input list. Each ligne is a dictionary containing a list of roads
    list_roads : (N,) list
        List of all appeared road names


    Returns
    -------
    vect : DataFrame
        The matix of all address in data.
    
    """
    vect = pd.DataFrame()
    length = len(list_roads) 
    for i in range(len(data)):
        vect_road = []
        found = 0
        j = 0
        roads = data[i]['roads']
        
        # stop when all roads in this address are found or all roads in 
        # list_roads are checked
        while((found < len(roads)) & (j < length)):
            if (list_roads[j] in roads):
                vect_road.append(1)
                found = found + 1
            else:
                vect_road.append(0)
            j = j + 1
        #data[i]['vect'] = vect_road
        vect = vect.append(pd.Series(vect_road), ignore_index=True)
    vect = vect.fillna(0)
    return vect


def n_cluster(vect, n_min = 5, n_max = 0, n_point = 5):
    """
    Choose the best number of cluster in given(or default) range
    
    Parameters
    ----------
    vect : DataFrame
        The matix of all address in data.
    n_min : int
        minimum number of clusters, default value is 5

    n_max: int 
        maximum number of clusters, default value is number of data/number of 
        point in each cluster
    n_point: 
        how many point in each cluster, default value is 5


    Returns
    -------
    n : int
        The number of cluster when clustering model will has the best 
        performance
    
    """
    distance = []
    k = []
    #when n_max isn't defined or n_max<n_min
    if (n_max < n_min):
        n_max = round(len(vect.columns)/n_point)
        
    #calcul the sum of distance inside cluster in each case
    for n_clusters in range(n_min,n_max):
        
        cls = KMeans(n_clusters).fit(vect)
        
        distance_sum = 0
        for i in range(n_clusters):
            group = cls.labels_ == i
            members = vect.iloc[group]
            for v in range(len(members)):
                distance_sum += hamming(np.array(members.iloc[v]), cls.cluster_centers_[i])
        distance.append(distance_sum)
        k.append(n_clusters)
        
    #find the best n in the defined range    
    diff = [distance[1:][i]-distance[:-1][i] for i in range(len(distance)-1)]
    n = k[diff.index(max(diff))]

    return n



# =============================================================================
# main function
# Perform a clustering based on either the location or characteristics of 
# bike stations.
# =============================================================================

#read json file
print("== 1/7 Read data")
file = input("Json file:")
#file = 'Brisbane_CityBike.json'
with open(file) as f:
    data = json.load(f)


# get the list of all appeared road names
print("== 2/7 Find all appeared road names")
list_roads = []
for i in range(len(data)):
    roads = data[i]['address']
    symbol_special=[' /  ',' (',')',' / ']
    for s in symbol_special:
        roads = roads.replace(s,'/')
    roads = re.split('/',roads)
    data[i]['roads'] = roads
    list_roads.extend(roads)
list_roads = list(set(list_roads))
list_roads = [x for x in list_roads if x != '']

#vectorize address
print("== 3/7 Vectorize address")
vect = vectorizer(data, list_roads)

#determine n_cluster
print("== 4/7 Determine n_cluster")
if (auto):
    n_cluster = n_cluster(vect, n_min, n_max, n_point)


print("== 5/7 Clustering")
model = KMeans(n_cluster)
df = pd.DataFrame.from_dict(data)
df['label'] = model.fit_predict(vect)

#find center of each cluster
print("== 6/7 Find centers")
centers = []
for i in range(n_cluster):
    lst = df[df.label == i]['roads'].tolist()
    cluster_data = [x for j in lst for x in j]
    center = max(set(cluster_data), key=cluster_data.count)
    if center == "":
        center = lst[0][0]
    centers.append(center)


#write result to csv files
print("== 7/7 Output")
with open('centers.txt', 'w') as f:
    i = 0
    for item in centers:
        f.write("%d\t%s\n" % (i,item))
        i = i+1

result = df.drop("roads", 1).sort_values("label")
result.to_csv("result.csv")