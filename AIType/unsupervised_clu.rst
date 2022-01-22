.. _unsupervised_clu:

Unsupervised - Clustering
===============

Template for Clustering


Content
-------

-  `1. Introduction <#0>`__
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

   -  `5.2. Hierarchial Clustering (Agglomerative Clustering) <#5.2>`__

      -  `5.2.1 Visualising the hierarchy <#5.2.1>`__

   -  `5.3. Affinity Propagation Clustering <#5.3>`__

      -  `5.3.1 Visualising the cluster <#5.2.1>`__

   -  `5.4. DBSCAN Clustering <#5.4>`__

      -  `5.3.1 Finding the right parameters <#5.4.1>`__

 # 1. Introduction

Clustering can serve to better understand the data through the lens of
categories learned from continuous variables. It also permits
automatically categorizing new objects according to the learned
criteria. Alternatively, clusters can be used to represent groups as
prototypes, using e.g. the midpoint of a cluster as the best
representatives of learned grouping.

In this jupyter notebook, we will look at the following clustering
techniques: 1. K-means 2. Hierarchical Clustering (Agglomerative
Clustering) 3. Affinity Propagation

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

    # load dataset
    dataset = read_csv('Data_MasterTemplate.csv',index_col=0)

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

    (4804, 30)



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
          <th>2000-01-03</th>
          <td>29.847</td>
          <td>35.477</td>
          <td>3.531</td>
          <td>26.650</td>
          <td>14.561</td>
          <td>21.582</td>
          <td>43.004</td>
          <td>16.984</td>
          <td>23.522</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.701</td>
          <td>16.747</td>
          <td>32.228</td>
          <td>20.159</td>
          <td>21.319</td>
          <td>5.841</td>
          <td>22.564</td>
          <td>NaN</td>
          <td>47.338</td>
          <td>21.713</td>
        </tr>
        <tr>
          <th>2000-01-04</th>
          <td>28.661</td>
          <td>34.134</td>
          <td>3.233</td>
          <td>26.610</td>
          <td>14.372</td>
          <td>21.582</td>
          <td>40.577</td>
          <td>17.041</td>
          <td>24.900</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.445</td>
          <td>16.122</td>
          <td>31.596</td>
          <td>19.890</td>
          <td>20.446</td>
          <td>5.766</td>
          <td>21.834</td>
          <td>NaN</td>
          <td>45.566</td>
          <td>20.907</td>
        </tr>
        <tr>
          <th>2000-01-05</th>
          <td>30.122</td>
          <td>33.959</td>
          <td>3.280</td>
          <td>28.474</td>
          <td>14.914</td>
          <td>22.049</td>
          <td>40.895</td>
          <td>17.228</td>
          <td>25.782</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.702</td>
          <td>16.416</td>
          <td>31.326</td>
          <td>20.086</td>
          <td>20.255</td>
          <td>5.753</td>
          <td>22.564</td>
          <td>NaN</td>
          <td>44.503</td>
          <td>21.097</td>
        </tr>
        <tr>
          <th>2000-01-06</th>
          <td>31.877</td>
          <td>33.959</td>
          <td>2.996</td>
          <td>28.553</td>
          <td>15.459</td>
          <td>22.903</td>
          <td>39.782</td>
          <td>17.210</td>
          <td>24.900</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.678</td>
          <td>16.973</td>
          <td>32.438</td>
          <td>20.122</td>
          <td>20.998</td>
          <td>5.964</td>
          <td>22.449</td>
          <td>NaN</td>
          <td>45.127</td>
          <td>20.527</td>
        </tr>
        <tr>
          <th>2000-01-07</th>
          <td>32.510</td>
          <td>34.434</td>
          <td>3.138</td>
          <td>29.382</td>
          <td>15.962</td>
          <td>23.306</td>
          <td>42.129</td>
          <td>18.342</td>
          <td>24.506</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.678</td>
          <td>18.123</td>
          <td>35.024</td>
          <td>20.922</td>
          <td>21.831</td>
          <td>6.663</td>
          <td>22.283</td>
          <td>NaN</td>
          <td>48.535</td>
          <td>21.052</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 30 columns</p>
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

Taking a look at the correlation. More detailed look at the data will be
performed after implementing the Dimensionality Reduction Models.

.. code:: ipython3

    # correlation
    correlation = dataset.corr()
    plt.figure(figsize=(15,15))
    plt.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x10fbb6bab70>




.. image:: output_19_1.png


 ## 4. Data Preparation

 ## 4.1. Data Cleaning Checking for the NAs in the rows, either drop
them or fill them with the mean of the column

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())


.. parsed-literal::

    Null Values = True


In this step we getting rid of the columns with more than 30% missing
values.

.. code:: ipython3

    missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

    missing_fractions.head(10)

    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

    dataset.drop(labels=drop_list, axis=1, inplace=True)
    dataset.shape




.. parsed-literal::

    (4804, 28)



Given that there are null values drop the rown contianing the null
values.

.. code:: ipython3

    # Fill the missing values with the last value available in the dataset.
    dataset=dataset.fillna(method='ffill')

    # Drop the rows containing NA
    #dataset= dataset.dropna(axis=0)
    # Fill na with 0
    #dataset.fillna('0')

    #Filling the NAs with the mean of the column.
    #dataset['col'] = dataset['col'].fillna(dataset['col'].mean())

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
          <th>MMM</th>
          <th>AXP</th>
          <th>AAPL</th>
          <th>BA</th>
          <th>CAT</th>
          <th>CVX</th>
          <th>CSCO</th>
          <th>KO</th>
          <th>DIS</th>
          <th>XOM</th>
          <th>...</th>
          <th>MSFT</th>
          <th>NKE</th>
          <th>PFE</th>
          <th>PG</th>
          <th>TRV</th>
          <th>UTX</th>
          <th>UNH</th>
          <th>VZ</th>
          <th>WMT</th>
          <th>WBA</th>
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
          <th>2000-01-03</th>
          <td>29.847</td>
          <td>35.477</td>
          <td>3.531</td>
          <td>26.65</td>
          <td>14.561</td>
          <td>21.582</td>
          <td>43.004</td>
          <td>16.984</td>
          <td>23.522</td>
          <td>23.862</td>
          <td>...</td>
          <td>38.135</td>
          <td>4.701</td>
          <td>16.747</td>
          <td>32.228</td>
          <td>20.159</td>
          <td>21.319</td>
          <td>5.841</td>
          <td>22.564</td>
          <td>47.338</td>
          <td>21.713</td>
        </tr>
        <tr>
          <th>2000-01-04</th>
          <td>28.661</td>
          <td>34.134</td>
          <td>3.233</td>
          <td>26.61</td>
          <td>14.372</td>
          <td>21.582</td>
          <td>40.577</td>
          <td>17.041</td>
          <td>24.900</td>
          <td>23.405</td>
          <td>...</td>
          <td>36.846</td>
          <td>4.445</td>
          <td>16.122</td>
          <td>31.596</td>
          <td>19.890</td>
          <td>20.446</td>
          <td>5.766</td>
          <td>21.834</td>
          <td>45.566</td>
          <td>20.907</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 28 columns</p>
    </div>



 ## 4.2. Data Transformation

In this step we preparing the data for the clustering.

.. code:: ipython3

    #Calculate average annual percentage return and volatilities over a theoretical one year period
    returns = dataset.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
    data=returns
    #format the data as a numpy array to feed into the K-Means algorithm
    #data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

All the variables should be on the same scale before applying PCA,
otherwise a feature with large values will dominate the result. Below I
use StandardScaler in scikit-learn to standardize the dataset’s features
onto unit scale (mean = 0 and variance = 1).

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
          <th>MMM</th>
          <td>0.059</td>
          <td>-1.010</td>
        </tr>
        <tr>
          <th>AXP</th>
          <td>-0.075</td>
          <td>1.115</td>
        </tr>
      </tbody>
    </table>
    </div>



The parameters to clusters are the indices and the variables used in the
clustering are the columns. Hence the data is in the right format to be
fed to the clustering algorithms

 # 5. Evaluate Algorithms and Models

We will look at the implementation and visualization of the following
clustering techniques.

1. KMeans
2. Hierarchial clustering
3. Affinity Propagation clustering

 ## 5.1. K-Means Clustering

k-Means is the most well-known clustering algorithm and was first
proposed by Stuart Lloyd at Bell Labs in 1957.

The algorithm finds K centroids and assigns each data point to exactly
one cluster with the goal of minimizing the within-cluster variance
(called inertia). It typically uses Euclidean distance but other metrics
can also be used. k-Means assumes that clusters are spherical and of
equal size and ignores the covariance among features.

The problem is computationally difficult (np-hard) because there are 𝐾N
ways to partition the N observations into K clusters. The standard
iterative algorithm delivers a local optimum for a given K and proceeds
as follows: 1. Randomly define K cluster centers and assign points to
nearest centroid 2. Repeat: 1. For each cluster, compute the centroid as
the average of the features 2. Assign each observation to the closest
centroid 3. Convergence: assignments (or within-cluster variation) don’t
change

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



In the preceding code, first, we import the KMeans package from
scikit-learn and initialize a k-means model. We then fit this model to
the data by using the .fit() function. This results in a set of labels
as the output. We can extract the labels by using the following code:

In the next step we extract the important parameters from the k-means
clustering

.. code:: ipython3

    centroids, assignments, inertia = k_means.cluster_centers_, k_means.labels_, k_means.inertia_

.. code:: ipython3

    #Extracting labels
    target_labels = k_means.predict(X)
    #Printing the labels
    target_labels




.. parsed-literal::

    array([5, 3, 1, 2, 2, 4, 3, 5, 4, 0, 3, 4, 0, 3, 5, 3, 5, 0, 4, 2, 0, 5,
           4, 4, 2, 0, 0, 4])



 ### 5.1.1. Finding optimal number of clusters

Typically, two metrics are used to evaluate a K-means model.

1. Sum of square errors (SSE) within clusters
2. Silhouette score.

SSE within clusters is derived by summing up the squared distance
between each data point and its closest centroid. The goal is to reduce
the error value. The intuition behind this is that we would want the
distance of each data point to be as close as possible to the centroid.
If the error is small, it would mean that the data points in the same
cluster are relatively similar. As the number of centroids (clusters)
increase, the error value will decrease. As such we would need to rely
on the next metric to ensure that we are not introducing too many
centroids (clusters) in the model.

Silhouette score is a measure of how similar the data point is to its
own cluster compared to other clusters. The value ranges from -1 (worst
score) to 1 (best score). A negative value would mean that data points
are wrongly clustered while values near 0 would mean that there are
overlapping clusters.

.. code:: ipython3

    distorsions = []
    max_loop=20
    for k in range(2, max_loop):
        kmeans_test = KMeans(n_clusters=k)
        kmeans_test.fit(X)
        distorsions.append(kmeans_test.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), distorsions)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)



.. image:: output_45_0.png


Silhouette score
^^^^^^^^^^^^^^^^

.. code:: ipython3

    from sklearn import metrics

    silhouette_score = []
    for k in range(2, max_loop):
            kmeans_test = KMeans(n_clusters=k,  random_state=10, n_init=10, n_jobs=-1)
            kmeans_test.fit(X)
            silhouette_score.append(metrics.silhouette_score(X, kmeans_test.labels_, random_state=10))
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), silhouette_score)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)



.. image:: output_47_0.png


From the first graph, Within Cluster SSE After K-Means Clustering, we
can see that as the number of clusters increase pass 3, the sum of
square of errors within clusters plateaus off. From the second graph,
Silhouette Score After K-Means Clustering, we can see that there are
various parts of the graph where a kink can be seen. Since there is not
much a difference in SSE after 7 clusters and that the drop in sihouette
score is quite significant between 14 clusters and 15 clusters, I would
use 14 clusters in my K-Means model below.

.. code:: ipython3

    k_means.labels_




.. parsed-literal::

    array([5, 3, 1, 2, 2, 4, 3, 5, 4, 0, 3, 4, 0, 3, 5, 3, 5, 0, 4, 2, 0, 5,
           4, 4, 2, 0, 0, 4])



 ### 5.1.2. Cluster Visualisation

Visualizing how your clusters are formed is no easy task when the number
of variables/dimensions in your dataset is very large. One of the
methods of visualising a cluster in two-dimensional space.

.. code:: ipython3

    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c = k_means.labels_, cmap ="rainbow", label = X.index)
    ax.set_title('k-Means')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)

    # zip joins x and y coordinates in pairs
    for x,y,name in zip(X.iloc[:,0],X.iloc[:,1],X.index):

        label = name

        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center

    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=11)




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x10fc2c216a0>]




.. image:: output_52_1.png


Checking Elements in each cluster

.. code:: ipython3

    cluster_label = pd.concat([pd.DataFrame(X.index), pd.DataFrame(k_means.labels_)],axis = 1)
    cluster_label.columns =['Company','Cluster']
    cluster_label.sort_values(by=['Cluster'])




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
          <th>Company</th>
          <th>Cluster</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>17</th>
          <td>MRK</td>
          <td>0</td>
        </tr>
        <tr>
          <th>25</th>
          <td>VZ</td>
          <td>0</td>
        </tr>
        <tr>
          <th>20</th>
          <td>PFE</td>
          <td>0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>XOM</td>
          <td>0</td>
        </tr>
        <tr>
          <th>26</th>
          <td>WMT</td>
          <td>0</td>
        </tr>
        <tr>
          <th>12</th>
          <td>IBM</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>AAPL</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>BA</td>
          <td>2</td>
        </tr>
        <tr>
          <th>4</th>
          <td>CAT</td>
          <td>2</td>
        </tr>
        <tr>
          <th>24</th>
          <td>UNH</td>
          <td>2</td>
        </tr>
        <tr>
          <th>19</th>
          <td>NKE</td>
          <td>2</td>
        </tr>
        <tr>
          <th>15</th>
          <td>JPM</td>
          <td>3</td>
        </tr>
        <tr>
          <th>13</th>
          <td>INTC</td>
          <td>3</td>
        </tr>
        <tr>
          <th>10</th>
          <td>GS</td>
          <td>3</td>
        </tr>
        <tr>
          <th>6</th>
          <td>CSCO</td>
          <td>3</td>
        </tr>
        <tr>
          <th>1</th>
          <td>AXP</td>
          <td>3</td>
        </tr>
        <tr>
          <th>22</th>
          <td>TRV</td>
          <td>4</td>
        </tr>
        <tr>
          <th>23</th>
          <td>UTX</td>
          <td>4</td>
        </tr>
        <tr>
          <th>27</th>
          <td>WBA</td>
          <td>4</td>
        </tr>
        <tr>
          <th>11</th>
          <td>HD</td>
          <td>4</td>
        </tr>
        <tr>
          <th>8</th>
          <td>DIS</td>
          <td>4</td>
        </tr>
        <tr>
          <th>5</th>
          <td>CVX</td>
          <td>4</td>
        </tr>
        <tr>
          <th>18</th>
          <td>MSFT</td>
          <td>4</td>
        </tr>
        <tr>
          <th>16</th>
          <td>MCD</td>
          <td>5</td>
        </tr>
        <tr>
          <th>14</th>
          <td>JNJ</td>
          <td>5</td>
        </tr>
        <tr>
          <th>21</th>
          <td>PG</td>
          <td>5</td>
        </tr>
        <tr>
          <th>7</th>
          <td>KO</td>
          <td>5</td>
        </tr>
        <tr>
          <th>0</th>
          <td>MMM</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 5.2. Hierarchical Clustering (Agglomerative Clustering)

Initially, each point is considered as a separate cluster, then it
recursively clusters the points together depending upon the distance
between them. The points are clustered in such a way that the distance
between points within a cluster is minimum and distance between the
cluster is maximum. Commonly used distance measures are Euclidean
distance, Manhattan distance or Mahalanobis distance. Unlike k-means
clustering, it is “bottom-up” approach.

Its primary advantage over other clustering methods is that you don’t
need to guess in advance how many clusters there might be. Agglomerate
Clustering first assigns each data point into its own cluster, and
gradually merges clusters until only one remains. It’s then up to the
user to choose a cutoff threshold and decide how many clusters are
present.

Python Tip: Though providing the number of clusters is not necessary but
Python provides an option of providing the same for easy and simple use.

While hierarchical clustering does not have hyperparameters like
k-Means, the measure of dissimilarity between clusters (as opposed to
individual data points) has an important impact on the clustering
result. The options differ as follows:

-  Single-link: distance between nearest neighbors of two clusters
-  Complete link: maximum distance between respective cluster members
-  Group average
-  Ward’s method: minimize within-cluster variance

The use of a distance metric makes hierarchical clustering sensitive to
scale:

.. code:: ipython3

    nclust = 4
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
    clust_labels1 = model.fit_predict(X)

.. code:: ipython3

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c =clust_labels1, cmap ="rainbow")
    ax.set_title('Hierarchial')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)

    # zip joins x and y coordinates in pairs
    for x,y,name in zip(X.iloc[:,0],X.iloc[:,1],X.index):

        label = name

        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center



.. image:: output_59_0.png


In this instance, the results between k-means and hierarchical
clustering were pretty similar. This is not always the case, however. In
general, the advantage of agglomerative hierarchical clustering is that
it tends to produce more accurate results. The downside is that
hierarchical clustering is more difficult to implement and more
time/resource consuming than k-means.

 ### 5.2.1. Visualisation : Building Hierarchy Graph/ Dendogram

The next step is to look for clusters of correlations using the
agglomerate hierarchical clustering technique. The hierarchy class has a
dendrogram method which takes the value returned by the linkage method
of the same class. The linkage method takes the dataset and the method
to minimize distances as parameters. We use ‘ward’ as the method since
it minimizes then variants of distances between the clusters.

Linkage does the actual clustering in one line of code, and returns a
list of the clusters joined in the format: Z=[stock_1, stock_2,
distance, sample_count]

There are also different options for the measurement of the distance.
The option we will choose is the average distance measurement, but
others are possible (ward, single, centroid, etc.).

.. code:: ipython3

    from scipy.cluster.hierarchy import dendrogram, linkage, ward

    #Calulate linkage
    Z= linkage(X, method='ward')
    Z[0]




.. parsed-literal::

    array([20.        , 25.        ,  0.06407423,  2.        ])



The best way to visualize an agglomerate clustering algorithm is through
a dendogram, which displays a cluster tree, the leaves being the
individual stocks and the root being the final single cluster. The
“distance” between each cluster is shown on the y-axis, and thus the
longer the branches are, the less correlated two clusters are.

.. code:: ipython3

    #Plot Dendogram
    plt.figure(figsize=(10, 7))
    plt.title("Stocks Dendograms")
    dendrogram(Z,labels = X.index)
    plt.show()



.. image:: output_65_0.png


Once one big cluster is formed, the longest vertical distance without
any horizontal line passing through it is selected and a horizontal line
is drawn through it. The number of vertical lines this newly created
horizontal line passes is equal to number of clusters. Then we select
the distance threshold to cut the dendrogram to obtain the selected
clustering level. The output is the cluster labelled for each row of
data. As expected from the dendrogram, a cut at 2.5 gives us 5 clusters.

.. code:: ipython3

    distance_threshold = 2.5
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])

    chosen_clusters['cluster'].unique()
    # array([4, 5, 2, 3, 1], dtype=int64)




.. parsed-literal::

    array([2, 3, 5, 4, 1], dtype=int64)



Cophenetic Correlation coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It’s important to get a sense of how well the clustering performs. One
measure is the Cophenetic Correlation Coefficient, c . This compares
(correlates) the actual pairwise distances of all your samples to those
implied by the hierarchical clustering. The closer c is to 1, the better
the clustering preserves the original distances. Generally c > 0.7 is
consistered a good cluster fit. Of course, other accuracy checks are
possible.

.. code:: ipython3

    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    import pylab
    c, coph_dists = cophenet(Z, pdist(X))
    c




.. parsed-literal::

    0.693558090937627



According to the dendogram above, the two most correlated stocks PAYX
and ADP. First, does this intuitively make sense? Doing a quick look at
PAYX and ADP, it seems like they are both in the business of taxes,
payroll, HR, retirement and insurance. So it makes sense that they would
be strongly correlated. Let’s plot them below to visually see how well
they correlate. In addition, let’s pick two stocks that are not well
correlated at all to compare to, say, NVDA and WDC.

 ### 5.2.2. Compare linkage types

Hierarchical clustering provides insight into degrees of similarity
among observations as it continues to merge data. A significant change
in the similarity metric from one merge to the next suggests a natural
clustering existed prior to this point. The dendrogram visualizes the
successive merges as a binary tree, displaying the individual data
points as leaves and the final merge as the root of the tree. It also
shows how the similarity monotonically decreases from bottom to top.
Hence, it is natural to select a clustering by cutting the dendrogram.

The following figure illustrates the dendrogram for the classic Iris
dataset with four classes and three features using the four different
distance metrics introduced above. It evaluates the fit of the
hierarchical clustering using the cophenetic correlation coefficient
that compares the pairwise distances among points and the cluster
similarity metric at which a pairwise merge occurred. A coefficient of 1
implies that closer points always merge earlier.

.. code:: ipython3

    methods = ['single', 'complete', 'average', 'ward']
    pairwise_distance = pdist(rescaledDataset)

.. code:: ipython3

    fig, axes = plt.subplots(figsize=(15, 8), nrows=2, ncols=2, sharex=True)
    axes = axes.flatten()
    for i, method in enumerate(methods):
        Z = linkage(X, method)
        c, coph_dists = cophenet(Z, pairwise_distance)
        dendrogram(Z, labels=X.index,
            orientation='top', leaf_rotation=0.,
            leaf_font_size=8., ax = axes[i])
        axes[i].set_title('Method: {} | Correlation: {:.2f}'.format(
                                                    method.capitalize(), c))

    fig.tight_layout()



.. image:: output_75_0.png


Different linkage methods produce different dendrogram ‘looks’ so that
we can not use this visualization to compare results across methods. In
addition, the Ward method that minimizes the within-cluster variance may
not properly reflect the change in variance but the total variance that
may be misleading. Instead, other quality metrics like the cophenetic
correlation or measures like inertia if aligned with the overall goal
are more appropriate.

The strengths of hierarchical clustering include that:

-  You do not need to specify the number of clusters but instead offers
   insight about potential clustering by means of an intuitive
   visualization.

-  It produces a hierarchy of clusters that can serve as a taxonomy.

-  It can be combined with k-means to reduce the number of items at the
   start of the agglomerative process.

The weaknesses include:

-  The high cost in terms of computation and memory because of the
   numerous similarity matrix updates.

-  Another downside is that all merges are final so that it does not
   achieve the global optimum.

-  Furthermore, the curse of dimensionality leads to difficulties with
   noisy, high-dimensional data.

 ## 5.3. Affinity Propagation

It does not require the number of cluster to be estimated and provided
before starting the algorithm. It makes no assumption regarding the
internal structure of the data points

The algorithm exchanges messages between pairs of data points until a
set of exemplars emerges, with each exemplar corresponding to a cluster.
The Affinity Propagation algorithm takes as input a real number s(k,k)
for each data point k — referred to as a “preference”. Data points with
large values for s(k,k) are more likely to be exemplars. The number of
clusters is influenced by the preference values and the message-passing
procedure.

.. code:: ipython3

    ap = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
    ap.fit(X)
    clust_labels2 = ap.predict(X)

.. code:: ipython3

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c =clust_labels2, cmap ="rainbow")
    ax.set_title('Affinity')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)

    # zip joins x and y coordinates in pairs
    for x,y,name in zip(X.iloc[:,0],X.iloc[:,1],X.index):

        label = name

        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center



.. image:: output_81_0.png


 ### 5.3.1 Cluster Visualisation

.. code:: ipython3

    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    n_clusters_ = len(cluster_centers_indices)

.. code:: ipython3

    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    no_clusters = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % no_clusters)
    # Plot exemplars

    X_temp=np.asarray(X)
    plt.close('all')
    plt.figure(1)
    plt.clf()

    fig = plt.figure(figsize=(8,6))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X_temp[cluster_centers_indices[k]]
        plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X_temp[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.show()


.. parsed-literal::

    Estimated number of clusters: 6



.. parsed-literal::

    <Figure size 432x288 with 0 Axes>



.. image:: output_84_2.png


.. code:: ipython3

    cluster_label = pd.concat([pd.DataFrame(X.index), pd.DataFrame(ap.labels_)],axis = 1)
    cluster_label.columns =['Company','Cluster']
    cluster_label.sort_values(by=['Cluster'])




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
          <th>Company</th>
          <th>Cluster</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>MMM</td>
          <td>0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>CVX</td>
          <td>0</td>
        </tr>
        <tr>
          <th>16</th>
          <td>MCD</td>
          <td>0</td>
        </tr>
        <tr>
          <th>14</th>
          <td>JNJ</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>AAPL</td>
          <td>1</td>
        </tr>
        <tr>
          <th>23</th>
          <td>UTX</td>
          <td>2</td>
        </tr>
        <tr>
          <th>22</th>
          <td>TRV</td>
          <td>2</td>
        </tr>
        <tr>
          <th>18</th>
          <td>MSFT</td>
          <td>2</td>
        </tr>
        <tr>
          <th>8</th>
          <td>DIS</td>
          <td>2</td>
        </tr>
        <tr>
          <th>11</th>
          <td>HD</td>
          <td>2</td>
        </tr>
        <tr>
          <th>15</th>
          <td>JPM</td>
          <td>3</td>
        </tr>
        <tr>
          <th>13</th>
          <td>INTC</td>
          <td>3</td>
        </tr>
        <tr>
          <th>6</th>
          <td>CSCO</td>
          <td>3</td>
        </tr>
        <tr>
          <th>1</th>
          <td>AXP</td>
          <td>3</td>
        </tr>
        <tr>
          <th>10</th>
          <td>GS</td>
          <td>3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>CAT</td>
          <td>4</td>
        </tr>
        <tr>
          <th>19</th>
          <td>NKE</td>
          <td>4</td>
        </tr>
        <tr>
          <th>3</th>
          <td>BA</td>
          <td>4</td>
        </tr>
        <tr>
          <th>24</th>
          <td>UNH</td>
          <td>4</td>
        </tr>
        <tr>
          <th>12</th>
          <td>IBM</td>
          <td>5</td>
        </tr>
        <tr>
          <th>26</th>
          <td>WMT</td>
          <td>5</td>
        </tr>
        <tr>
          <th>9</th>
          <td>XOM</td>
          <td>5</td>
        </tr>
        <tr>
          <th>7</th>
          <td>KO</td>
          <td>5</td>
        </tr>
        <tr>
          <th>17</th>
          <td>MRK</td>
          <td>5</td>
        </tr>
        <tr>
          <th>20</th>
          <td>PFE</td>
          <td>5</td>
        </tr>
        <tr>
          <th>21</th>
          <td>PG</td>
          <td>5</td>
        </tr>
        <tr>
          <th>25</th>
          <td>VZ</td>
          <td>5</td>
        </tr>
        <tr>
          <th>27</th>
          <td>WBA</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>
