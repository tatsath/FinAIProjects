.. _AlgoTrd_clust:


Pairs Trading- finding pairs based on Clustering
================================================

In this case study, we will use clustering methods to select pairs for a
pairs trading strategy.

Content
-------

-  `1. Problem Definition <#0>`__
-  `2. Getting Started - Load Libraries and Dataset <#1>`__

   -  `2.1. Load Libraries <#1.1>`__
   -  `2.2. Load Dataset <#1.2>`__

-  `3. Exploratory Data Analysis <#2>`__

   -  `3.1 Descriptive Statistics <#2.1>`__
   -  `3.2. Data Visualisation <#2.2>`__

-  `4. Data Preparation <#3>`__

   -  `4.1 Data Cleaning <#3.1>`__
   -  `4.3.Data Transformation <#3.2>`__

-  `5.Evaluate Algorithms and Models <#5>`__

   -  `5.1. k-Means Clustering <#5.1>`__

      -  `5.1.1 Finding right number of clusters <#5.1.1>`__
      -  `5.1.2 Clustering and Visualization <#5.1.2>`__

   -  `5.2. Hierarchial Clustering (Agglomerative Clustering) <#5.2>`__

      -  `5.2.1. Building Hierarchy Graph/ Dendogram <#5.2.1>`__
      -  `5.2.2. Clustering and Visualization <#5.2.1>`__

   -  `5.3. Affinity Propagation Clustering <#5.3>`__

      -  `5.3.1 Visualising the cluster <#5.2.1>`__

   -  `5.4. Cluster Evaluation <#5.4>`__

-  `6.Pair Selection <#6>`__

   -  `6.1 Cointegration and Pair Selection Function <#6.1>`__
   -  `6.2. Pair Visualization <#6.2>`__

 # 1. Problem Definition

Our goal in this case study is to perform clustering analysis on the
stocks of S&P500 and come up with pairs for a pairs trading strategy.

The data of the stocks of S&P 500, obtained using pandas_datareader from
yahoo finance. It includes price data from 2018 onwards.

 # 2. Getting Started- Loading the data and python packages

 ## 2.1. Loading the python packages

.. code:: ipython3

    # Load libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas import read_csv, set_option
    from pandas.plotting import scatter_matrix
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    import datetime
    import pandas_datareader as dr

    #Import Model Packages
    from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation, DBSCAN
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn import cluster, covariance, manifold


    #Other Helper Packages and functions
    import matplotlib.ticker as ticker
    from itertools import cycle

 ## 2.2. Loading the Data

.. code:: ipython3

    #The data already obtained from yahoo finance is imported.
    dataset = read_csv('SP500Data.csv',index_col=0)

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

.. code:: ipython3

    type(dataset)




.. parsed-literal::

    pandas.core.frame.DataFrame



 # 3. Exploratory Data Analysis

 ## 3.1. Descriptive Statistics

.. code:: ipython3

    # shape
    dataset.shape




.. parsed-literal::

    (448, 502)



.. code:: ipython3

    # peek at data
    set_option('display.width', 100)
    dataset.head(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ABT</th>
          <th>ABBV</th>
          <th>ABMD</th>
          <th>ACN</th>
          <th>ATVI</th>
          <th>ADBE</th>
          <th>AMD</th>
          <th>AAP</th>
          <th>AES</th>
          <th>AMG</th>
          <th>...</th>
          <th>WLTW</th>
          <th>WYNN</th>
          <th>XEL</th>
          <th>XRX</th>
          <th>XLNX</th>
          <th>XYL</th>
          <th>YUM</th>
          <th>ZBH</th>
          <th>ZION</th>
          <th>ZTS</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2018-01-02</th>
          <td>58.790001</td>
          <td>98.410004</td>
          <td>192.490005</td>
          <td>153.839996</td>
          <td>64.309998</td>
          <td>177.699997</td>
          <td>10.98</td>
          <td>106.089996</td>
          <td>10.88</td>
          <td>203.039993</td>
          <td>...</td>
          <td>146.990005</td>
          <td>164.300003</td>
          <td>47.810001</td>
          <td>29.370001</td>
          <td>67.879997</td>
          <td>68.070000</td>
          <td>81.599998</td>
          <td>124.059998</td>
          <td>50.700001</td>
          <td>71.769997</td>
        </tr>
        <tr>
          <th>2018-01-03</th>
          <td>58.919998</td>
          <td>99.949997</td>
          <td>195.820007</td>
          <td>154.550003</td>
          <td>65.309998</td>
          <td>181.039993</td>
          <td>11.55</td>
          <td>107.050003</td>
          <td>10.87</td>
          <td>202.119995</td>
          <td>...</td>
          <td>149.740005</td>
          <td>162.520004</td>
          <td>47.490002</td>
          <td>29.330000</td>
          <td>69.239998</td>
          <td>68.900002</td>
          <td>81.529999</td>
          <td>124.919998</td>
          <td>50.639999</td>
          <td>72.099998</td>
        </tr>
        <tr>
          <th>2018-01-04</th>
          <td>58.820000</td>
          <td>99.379997</td>
          <td>199.250000</td>
          <td>156.380005</td>
          <td>64.660004</td>
          <td>183.220001</td>
          <td>12.12</td>
          <td>111.000000</td>
          <td>10.83</td>
          <td>198.539993</td>
          <td>...</td>
          <td>151.259995</td>
          <td>163.399994</td>
          <td>47.119999</td>
          <td>29.690001</td>
          <td>70.489998</td>
          <td>69.360001</td>
          <td>82.360001</td>
          <td>124.739998</td>
          <td>50.849998</td>
          <td>72.529999</td>
        </tr>
        <tr>
          <th>2018-01-05</th>
          <td>58.990002</td>
          <td>101.110001</td>
          <td>202.320007</td>
          <td>157.669998</td>
          <td>66.370003</td>
          <td>185.339996</td>
          <td>11.88</td>
          <td>112.180000</td>
          <td>10.87</td>
          <td>199.470001</td>
          <td>...</td>
          <td>152.229996</td>
          <td>164.490005</td>
          <td>46.790001</td>
          <td>29.910000</td>
          <td>74.150002</td>
          <td>69.230003</td>
          <td>82.839996</td>
          <td>125.980003</td>
          <td>50.869999</td>
          <td>73.360001</td>
        </tr>
        <tr>
          <th>2018-01-08</th>
          <td>58.820000</td>
          <td>99.489998</td>
          <td>207.800003</td>
          <td>158.929993</td>
          <td>66.629997</td>
          <td>185.039993</td>
          <td>12.28</td>
          <td>111.389999</td>
          <td>10.87</td>
          <td>200.529999</td>
          <td>...</td>
          <td>151.410004</td>
          <td>162.300003</td>
          <td>47.139999</td>
          <td>30.260000</td>
          <td>74.639999</td>
          <td>69.480003</td>
          <td>82.980003</td>
          <td>126.220001</td>
          <td>50.619999</td>
          <td>74.239998</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 502 columns</p>
    </div>



.. code:: ipython3

    # describe data
    set_option('precision', 3)
    dataset.describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>MMM</th>
          <th>AXP</th>
          <th>AAPL</th>
          <th>BA</th>
          <th>CAT</th>
          <th>CVX</th>
          <th>CSCO</th>
          <th>KO</th>
          <th>DIS</th>
          <th>DWDP</th>
          <th>...</th>
          <th>NKE</th>
          <th>PFE</th>
          <th>PG</th>
          <th>TRV</th>
          <th>UTX</th>
          <th>UNH</th>
          <th>VZ</th>
          <th>V</th>
          <th>WMT</th>
          <th>WBA</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>363.000</td>
          <td>...</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
          <td>2741.000</td>
          <td>4804.000</td>
          <td>4804.000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>86.769</td>
          <td>49.659</td>
          <td>49.107</td>
          <td>85.482</td>
          <td>56.697</td>
          <td>61.735</td>
          <td>21.653</td>
          <td>24.984</td>
          <td>46.368</td>
          <td>64.897</td>
          <td>...</td>
          <td>23.724</td>
          <td>20.737</td>
          <td>49.960</td>
          <td>55.961</td>
          <td>62.209</td>
          <td>64.418</td>
          <td>27.193</td>
          <td>53.323</td>
          <td>50.767</td>
          <td>41.697</td>
        </tr>
        <tr>
          <th>std</th>
          <td>53.942</td>
          <td>22.564</td>
          <td>55.020</td>
          <td>79.085</td>
          <td>34.663</td>
          <td>31.714</td>
          <td>10.074</td>
          <td>10.611</td>
          <td>32.733</td>
          <td>5.768</td>
          <td>...</td>
          <td>20.988</td>
          <td>7.630</td>
          <td>19.769</td>
          <td>34.644</td>
          <td>32.627</td>
          <td>62.920</td>
          <td>11.973</td>
          <td>37.647</td>
          <td>17.040</td>
          <td>19.937</td>
        </tr>
        <tr>
          <th>min</th>
          <td>25.140</td>
          <td>8.713</td>
          <td>0.828</td>
          <td>17.463</td>
          <td>9.247</td>
          <td>17.566</td>
          <td>6.842</td>
          <td>11.699</td>
          <td>11.018</td>
          <td>49.090</td>
          <td>...</td>
          <td>2.595</td>
          <td>8.041</td>
          <td>16.204</td>
          <td>13.287</td>
          <td>14.521</td>
          <td>5.175</td>
          <td>11.210</td>
          <td>9.846</td>
          <td>30.748</td>
          <td>17.317</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>51.192</td>
          <td>34.079</td>
          <td>3.900</td>
          <td>37.407</td>
          <td>26.335</td>
          <td>31.820</td>
          <td>14.910</td>
          <td>15.420</td>
          <td>22.044</td>
          <td>62.250</td>
          <td>...</td>
          <td>8.037</td>
          <td>15.031</td>
          <td>35.414</td>
          <td>29.907</td>
          <td>34.328</td>
          <td>23.498</td>
          <td>17.434</td>
          <td>18.959</td>
          <td>38.062</td>
          <td>27.704</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>63.514</td>
          <td>42.274</td>
          <td>23.316</td>
          <td>58.437</td>
          <td>53.048</td>
          <td>56.942</td>
          <td>18.578</td>
          <td>20.563</td>
          <td>29.521</td>
          <td>66.586</td>
          <td>...</td>
          <td>14.147</td>
          <td>18.643</td>
          <td>46.735</td>
          <td>39.824</td>
          <td>55.715</td>
          <td>42.924</td>
          <td>21.556</td>
          <td>45.207</td>
          <td>42.782</td>
          <td>32.706</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>122.906</td>
          <td>66.816</td>
          <td>84.007</td>
          <td>112.996</td>
          <td>76.488</td>
          <td>91.688</td>
          <td>24.650</td>
          <td>34.927</td>
          <td>75.833</td>
          <td>69.143</td>
          <td>...</td>
          <td>36.545</td>
          <td>25.403</td>
          <td>68.135</td>
          <td>80.767</td>
          <td>92.557</td>
          <td>73.171</td>
          <td>38.996</td>
          <td>76.966</td>
          <td>65.076</td>
          <td>58.165</td>
        </tr>
        <tr>
          <th>max</th>
          <td>251.981</td>
          <td>112.421</td>
          <td>231.260</td>
          <td>411.110</td>
          <td>166.832</td>
          <td>128.680</td>
          <td>63.698</td>
          <td>50.400</td>
          <td>117.973</td>
          <td>75.261</td>
          <td>...</td>
          <td>85.300</td>
          <td>45.841</td>
          <td>98.030</td>
          <td>146.564</td>
          <td>141.280</td>
          <td>286.330</td>
          <td>60.016</td>
          <td>150.525</td>
          <td>107.010</td>
          <td>90.188</td>
        </tr>
      </tbody>
    </table>
    <p>8 rows × 30 columns</p>
    </div>



 ## 3.2. Data Visualization

We will take a detailed look into the visualization post clustering.

 ## 4. Data Preparation

 ## 4.1. Data Cleaning We check for the NAs in the rows, either drop
them or fill them with the mean of the column.

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())


.. parsed-literal::

    Null Values = True


Getting rid of the columns with more than 30% missing values.

.. code:: ipython3

    missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

    missing_fractions.head(10)

    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

    dataset.drop(labels=drop_list, axis=1, inplace=True)
    dataset.shape




.. parsed-literal::

    (448, 498)



Given that there are null values drop the rown contianing the null
values.

.. code:: ipython3

    # Fill the missing values with the last value available in the dataset.
    dataset=dataset.fillna(method='ffill')
    dataset.head(2)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ABT</th>
          <th>ABBV</th>
          <th>ABMD</th>
          <th>ACN</th>
          <th>ATVI</th>
          <th>ADBE</th>
          <th>AMD</th>
          <th>AAP</th>
          <th>AES</th>
          <th>AMG</th>
          <th>...</th>
          <th>WLTW</th>
          <th>WYNN</th>
          <th>XEL</th>
          <th>XRX</th>
          <th>XLNX</th>
          <th>XYL</th>
          <th>YUM</th>
          <th>ZBH</th>
          <th>ZION</th>
          <th>ZTS</th>
        </tr>
        <tr>
          <th>Date</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2018-01-02</th>
          <td>58.790001</td>
          <td>98.410004</td>
          <td>192.490005</td>
          <td>153.839996</td>
          <td>64.309998</td>
          <td>177.699997</td>
          <td>10.98</td>
          <td>106.089996</td>
          <td>10.88</td>
          <td>203.039993</td>
          <td>...</td>
          <td>146.990005</td>
          <td>164.300003</td>
          <td>47.810001</td>
          <td>29.370001</td>
          <td>67.879997</td>
          <td>68.070000</td>
          <td>81.599998</td>
          <td>124.059998</td>
          <td>50.700001</td>
          <td>71.769997</td>
        </tr>
        <tr>
          <th>2018-01-03</th>
          <td>58.919998</td>
          <td>99.949997</td>
          <td>195.820007</td>
          <td>154.550003</td>
          <td>65.309998</td>
          <td>181.039993</td>
          <td>11.55</td>
          <td>107.050003</td>
          <td>10.87</td>
          <td>202.119995</td>
          <td>...</td>
          <td>149.740005</td>
          <td>162.520004</td>
          <td>47.490002</td>
          <td>29.330000</td>
          <td>69.239998</td>
          <td>68.900002</td>
          <td>81.529999</td>
          <td>124.919998</td>
          <td>50.639999</td>
          <td>72.099998</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 498 columns</p>
    </div>



 ## 4.2. Data Transformation

For the purpose of clustering, we will be using annual returns and
variance as the variables as they are the indicators of the stock
performance and its volatility. Let us prepare the return and volatility
variables from the data.

.. code:: ipython3

    #Calculate average annual percentage return and volatilities over a theoretical one year period
    returns = dataset.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
    data=returns
    #format the data as a numpy array to feed into the K-Means algorithm
    #data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

All the variables should be on the same scale before applying
clustering, otherwise a feature with large values will dominate the
result. We use StandardScaler in sklearn to standardize the dataset’s
features onto unit scale (mean = 0 and variance = 1).

.. code:: ipython3

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(data),columns = data.columns, index = data.index)
    # summarize transformed data
    rescaledDataset.head(2)
    X=rescaledDataset
    X.head(2)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Returns</th>
          <th>Volatility</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>ABT</th>
          <td>0.794067</td>
          <td>-0.702741</td>
        </tr>
        <tr>
          <th>ABBV</th>
          <td>-0.927603</td>
          <td>0.794867</td>
        </tr>
      </tbody>
    </table>
    </div>



The parameters to clusters are the indices and the variables used in the
clustering are the columns. Hence the data is in the right format to be
fed to the clustering algorithms

 # 5. Evaluate Algorithms and Models

We will look at the following models:

1. KMeans
2. Hierarchical Clustering (Agglomerative Clustering)
3. Affinity Propagation

 ## 5.1. K-Means Clustering

 ### 5.1.1. Finding optimal number of clusters

In this step we look at the following metrices:

1. Sum of square errors (SSE) within clusters
2. Silhouette score.

.. code:: ipython3

    distorsions = []
    max_loop=20
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), distorsions)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)



.. image:: output_37_0.png


Inspecting the sum of squared errors chart, it appears the elbow “kink”
occurs 5 or 6 clusters for this data. Certainly, we can see that as the
number of clusters increase pass 6, the sum of square of errors within
clusters plateaus off.

Silhouette score
^^^^^^^^^^^^^^^^

.. code:: ipython3

    from sklearn import metrics

    silhouette_score = []
    for k in range(2, max_loop):
            kmeans = KMeans(n_clusters=k,  random_state=10, n_init=10, n_jobs=-1)
            kmeans.fit(X)
            silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), silhouette_score)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)



.. image:: output_40_0.png


From the silhouette score chart, we can see that there are various parts
of the graph where a kink can be seen. Since there is not much a
difference in SSE after 6 clusters, we would prefer 6 clusters in the
K-means model.

 ### 5.1.2. Clustering and Visualisation

Let us build the k-means model with six clusters and visualize the
results.

.. code:: ipython3

    nclust=6

.. code:: ipython3

    #Fit with k-means
    k_means = cluster.KMeans(n_clusters=nclust)
    k_means.fit(X)




.. parsed-literal::

    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=6, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=None, tol=0.0001, verbose=0)



.. code:: ipython3

    #Extracting labels
    target_labels = k_means.predict(X)

Visualizing how your clusters are formed is no easy task when the number
of variables/dimensions in your dataset is very large. One of the
methods of visualising a cluster in two-dimensional space.

.. code:: ipython3

    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c = k_means.labels_, cmap ="rainbow", label = X.index)
    ax.set_title('k-Means results')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)

    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=11)




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x1532b4d6710>]




.. image:: output_48_1.png


Let us check the elements of the clusters

.. code:: ipython3

    # show number of stocks in each cluster
    clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    # clustered stock with its cluster label
    clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series = clustered_series[clustered_series != -1]

    plt.figure(figsize=(12,7))
    plt.barh(
        range(len(clustered_series.value_counts())), # cluster labels, y axis
        clustered_series.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()



.. image:: output_50_0.png


The number of stocks in a cluster range from around 40 to 120. Although,
the distribution is not equal, we have significant number of stocks in
each cluster.

 ## 5.2. Hierarchical Clustering (Agglomerative Clustering)

In the first step we look at the hierarchy graph and check for the
number of clusters

 ### 5.2.1. Building Hierarchy Graph/ Dendogram

The hierarchy class has a dendrogram method which takes the value
returned by the linkage method of the same class. The linkage method
takes the dataset and the method to minimize distances as parameters. We
use ‘ward’ as the method since it minimizes then variants of distances
between the clusters.

.. code:: ipython3

    from scipy.cluster.hierarchy import dendrogram, linkage, ward

    #Calulate linkage
    Z= linkage(X, method='ward')
    Z[0]




.. parsed-literal::

    array([3.30000000e+01, 3.14000000e+02, 3.62580431e-03, 2.00000000e+00])



The best way to visualize an agglomerate clustering algorithm is through
a dendogram, which displays a cluster tree, the leaves being the
individual stocks and the root being the final single cluster. The
“distance” between each cluster is shown on the y-axis, and thus the
longer the branches are, the less correlated two clusters are.

.. code:: ipython3

    #Plot Dendogram
    plt.figure(figsize=(10, 7))
    plt.title("Stocks Dendrograms")
    dendrogram(Z,labels = X.index)
    plt.show()



.. image:: output_58_0.png


Once one big cluster is formed, the longest vertical distance without
any horizontal line passing through it is selected and a horizontal line
is drawn through it. The number of vertical lines this newly created
horizontal line passes is equal to number of clusters. Then we select
the distance threshold to cut the dendrogram to obtain the selected
clustering level. The output is the cluster labelled for each row of
data. As expected from the dendrogram, a cut at 13 gives us four
clusters.

.. code:: ipython3

    distance_threshold = 13
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
    chosen_clusters['cluster'].unique()




.. parsed-literal::

    array([1, 4, 3, 2], dtype=int64)



 ### 5.2.2. Clustering and Visualisation

.. code:: ipython3

    nclust = 4
    hc = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
    clust_labels1 = hc.fit_predict(X)

.. code:: ipython3

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c =clust_labels1, cmap ="rainbow")
    ax.set_title('Hierarchical Clustering')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x1fa81d717f0>




.. image:: output_63_1.png


Similar to the plot of k-means clustering, we see that there are some
distinct clusters separated by different colors.

 ## 5.3. Affinity Propagation

.. code:: ipython3

    ap = AffinityPropagation()
    ap.fit(X)
    clust_labels2 = ap.predict(X)

.. code:: ipython3

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c =clust_labels2, cmap ="rainbow")
    ax.set_title('Affinity')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x1498fae5390>




.. image:: output_67_1.png


Similar to the plot of k-means clustering, we see that there are some
distinct clusters separated by different colors.

 ### 5.3.1 Cluster Visualisation

.. code:: ipython3

    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_

.. code:: ipython3

    no_clusters = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % no_clusters)
    # Plot exemplars

    X_temp=np.asarray(X)
    plt.close('all')
    plt.figure(1)
    plt.clf()

    fig = plt.figure(figsize=(8,6))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(no_clusters), colors):
        class_members = labels == k
        cluster_center = X_temp[cluster_centers_indices[k]]
        plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X_temp[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.show()


.. parsed-literal::

    Estimated number of clusters: 27



.. parsed-literal::

    <Figure size 432x288 with 0 Axes>



.. image:: output_71_2.png


.. code:: ipython3

    # show number of stocks in each cluster
    clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    # clustered stock with its cluster label
    clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    clustered_series_ap = clustered_series_ap[clustered_series != -1]

    plt.figure(figsize=(12,7))
    plt.barh(
        range(len(clustered_series_ap.value_counts())), # cluster labels, y axis
        clustered_series_ap.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()



.. image:: output_72_0.png


 ## 5.4. Cluster Evaluation

If the ground truth labels are not known, evaluation must be performed
using the model itself. The Silhouette Coefficient
(sklearn.metrics.silhouette_score) is an example of such an evaluation,
where a higher Silhouette Coefficient score relates to a model with
better defined clusters. The Silhouette Coefficient is defined for each
sample and is composed of two scores:

.. code:: ipython3

    from sklearn import metrics
    print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
    print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
    print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))


.. parsed-literal::

    km 0.3350720873411941
    hc 0.3432149515640865
    ap 0.3450647315156527


Given the affinity propagation performs the best, we go ahead with the
affinity propagation and use 27 clusters as specified by this clustering
method

Visualising the return within a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The understand the intuition behind clustering, let us visualize the
results of the clusters.

.. code:: ipython3

    # all stock with its cluster label (including -1)
    clustered_series = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
    # clustered stock with its cluster label
    clustered_series_all = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
    clustered_series = clustered_series[clustered_series != -1]

.. code:: ipython3

    # get the number of stocks in each cluster
    counts = clustered_series_ap.value_counts()

    # let's visualize some clusters
    cluster_vis_list = list(counts[(counts<25) & (counts>1)].index)[::-1]
    cluster_vis_list




.. parsed-literal::

    [11, 25, 16, 20, 15, 2, 0, 5, 19, 17, 22, 21, 24, 10, 9, 13]



.. code:: ipython3

    CLUSTER_SIZE_LIMIT = 9999
    counts = clustered_series.value_counts()
    ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
    print ("Clusters formed: %d" % len(ticker_count_reduced))
    print ("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())


.. parsed-literal::

    Clusters formed: 26
    Pairs to evaluate: 12166


.. code:: ipython3

    # plot a handful of the smallest clusters
    plt.figure(figsize=(12,7))
    cluster_vis_list[0:min(len(cluster_vis_list), 4)]




.. parsed-literal::

    [11, 25, 16, 20]




.. parsed-literal::

    <Figure size 864x504 with 0 Axes>


.. code:: ipython3

    for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
        tickers = list(clustered_series[clustered_series==clust].index)
        means = np.log(dataset.loc[:"2018-02-01", tickers].mean())
        data = np.log(dataset.loc[:"2018-02-01", tickers]).sub(means)
        data.plot(title='Stock Time Series for Cluster %d' % clust)
    plt.show()



.. image:: output_83_0.png



.. image:: output_83_1.png



.. image:: output_83_2.png



.. image:: output_83_3.png


Looking at the charts above, across all the clusters with small number
of stocks, we see similar movement of the stocks under different
clusters, which corroborates the effectiveness of the clustering
technique.

 # 6. Pairs Selection

 ## 6.1. Cointegration and Pair Selection Function

.. code:: ipython3

    def find_cointegrated_pairs(data, significance=0.05):
        # This function is from https://www.quantopian.com/lectures/introduction-to-pairs-trading
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i in range(1):
            for j in range(i+1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < significance:
                    pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs

.. code:: ipython3

    from statsmodels.tsa.stattools import coint
    cluster_dict = {}
    for i, which_clust in enumerate(ticker_count_reduced.index):
        tickers = clustered_series[clustered_series == which_clust].index
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
            dataset[tickers]
        )
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs

.. code:: ipython3

    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])

.. code:: ipython3

    print ("Number of pairs found : %d" % len(pairs))
    print ("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))


.. parsed-literal::

    Number of pairs found : 32
    In those pairs, there are 47 unique tickers.


.. code:: ipython3

    pairs




.. parsed-literal::

    [('AOS', 'FITB'),
     ('AOS', 'ZION'),
     ('AIG', 'TEL'),
     ('ABBV', 'BWA'),
     ('AFL', 'ARE'),
     ('AFL', 'ED'),
     ('AFL', 'MMC'),
     ('AFL', 'WM'),
     ('ACN', 'EQIX'),
     ('A', 'WAT'),
     ('ADBE', 'ADI'),
     ('ADBE', 'CDNS'),
     ('ADBE', 'VFC'),
     ('ABT', 'AZO'),
     ('ABT', 'CHD'),
     ('ABT', 'IQV'),
     ('ABT', 'WELL'),
     ('ALL', 'GL'),
     ('MO', 'CCL'),
     ('ALB', 'CTL'),
     ('ALB', 'FANG'),
     ('ALB', 'EOG'),
     ('ALB', 'HP'),
     ('ALB', 'NOV'),
     ('ALB', 'PVH'),
     ('ALB', 'TPR'),
     ('ADSK', 'ULTA'),
     ('ADSK', 'XLNX'),
     ('AAL', 'FCX'),
     ('CMG', 'EW'),
     ('CMG', 'KEYS'),
     ('XEC', 'DXC')]



 ## 6.2. Pair Visualization

.. code:: ipython3

    from sklearn.manifold import TSNE
    import matplotlib.cm as cm
    stocks = np.unique(pairs)
    X_df = pd.DataFrame(index=X.index, data=X).T

.. code:: ipython3

    in_pairs_series = clustered_series.loc[stocks]
    stocks = list(np.unique(pairs))
    X_pairs = X_df.T.loc[stocks]

.. code:: ipython3

    X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)

.. code:: ipython3

    plt.figure(1, facecolor='white',figsize=(16,8))
    plt.clf()
    plt.axis('off')
    for pair in pairs:
        #print(pair[0])
        ticker1 = pair[0]
        loc1 = X_pairs.index.get_loc(pair[0])
        x1, y1 = X_tsne[loc1, :]
        #print(ticker1, loc1)

        ticker2 = pair[0]
        loc2 = X_pairs.index.get_loc(pair[1])
        x2, y2 = X_tsne[loc2, :]

        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=in_pairs_series.values, cmap=cm.Paired)
    plt.title('T-SNE Visualization of Validated Pairs');

    # zip joins x and y coordinates in pairs
    for x,y,name in zip(X_tsne[:,0],X_tsne[:,1],X_pairs.index):

        label = name

        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=11)





.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x14901bdf7f0>]




.. image:: output_96_1.png


**Conclusion**

The clustering techniques do not directly help in stock trend
prediction. However, they can be effectively used in portfolio
construction for finding the right pairs, which eventually help in risk
mitigation and one can achieve superior risk adjusted returns.

We showed the approaches to finding the appropriate number of clusters
in k-means and built a hierarchy graph in hierarchical clustering. A
next step from this case study would be to explore and backtest various
long/short trading strategies with pairs of stocks from the groupings of
stocks.

Clustering can effectively be used for dividing stocks into groups with
“similar characteristics” for many other kinds of trading strategies and
can help in portfolio construction to ensure we choose a universe of
stocks with sufficient diversification between them.
