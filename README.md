# Test_DS

    <span class="hljs-variable">@author:</span> Hong YU
    `</pre>

    * * *

    ## Description:

    ### Data :

    Static geographical information of CityBike‘s stations in Brisbane (“Brisbane_CityBike.json”)

    ### Instructions:

    Propose a script to perform a clustering based on either the location or characteristics of bike stations. 
    The code should be launched by a UNIX like command. It should be developed as close as a real industrialized process (well documented, parameters easily modifiable, readable logging, etc.).

    ### Languages :

    Python3

    ### Deliveries :

*   **Code**: A zip file containing all your code or a link to a Github repository
*   **Readme**: command line to launch the job, and short description of your program
*   **Output**: directory with the clustering result

    * * *

    ## clustering.py

    Perform a **k-means** clustering based on the address of bike stations.
    Firstly, separate road names in each addree of station.
    Then find out all appeared road names in this json file.
    Before clustering, translate the address to a one-hot vector.
    The number of cluters can be:

*   defined by user: **`n_cluster`**
*   determined automatically when **`auto=True`**:

        *   in range of given minimum and maximum number of cluster  **`n_min, n_max`**
    *   in range of minimum **`n_min`** and maximum number of cluster where the maximum is the largest number possible according to the defined number of data in each cluster **`n_point`**
    *   in default range (1, number of data/5)*   **n_cluster** is choosen when the sum of inner-cluster distance comes to a low point and foloowed by the largest increase

    ### Input:

*   path of json file

    ### Usage:

    <pre>`<span class="hljs-keyword">python3</span> clustering.<span class="hljs-keyword">py</span>

### Functions:

##### - vectorize

Creat one-hot vector for each address in the data

###### Parameters

`data` : Input list. Each ligne is a dictionary containing a list of roads
`list_roads` : List of all appeared road names

###### Returns

`vect` : The matix of all address in data.

##### - n_cluster

Choose the best number of cluster in given(or default) range

###### Parameters

`vect` : The matix of all address in data.
`n_min` : Minimum number of clusters, default value is 1
`n_max`:  Maximum number of clusters, default value is number of data/number of point in each cluster
`n_point`: How many point in each cluster, default value is 5

###### Returns

`n` : The number of cluster when clustering model will has the best performance