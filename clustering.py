# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:22:07 2019

@author: Hong
"""

def vectorizer(data, list_roads):
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
# =============================================================================
def main(auto=True, n_cluster = 5, n_min = 5, n_max = 0, n_point = 5):
    #read json file
    file = input("Json file:")
    file = 'Brisbane_CityBike.json'
    with open(file) as f:
        data = json.load(f)
    
    # get the list of all appeared road names
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
    vect = vectorizer(data, list_roads)
    
    if (auto):
        n_cluster = n_cluster(vect, n_min, n_max, n_point)
    model = KMeans(n_clusters=5)
    data2 = pd.DataFrame.from_dict(data)
    data2['label'] = model.fit_predict(vect)