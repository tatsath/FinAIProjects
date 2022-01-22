.. _unsupervised_dim:

Unsupervised - Dimensionality Reduction
===============

Template for Dimensionality reduction

In this notebook, we will learn about fundamental dimensionality
reduction techniques that will help us to summarize the information
content of a dataset by transforming it onto a new feature subspace of
lower dimensionality than the original one.

The details of the dimensionality Reduction Methodology from scikit
Learn can be found under:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition

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

-  `5.Evaluate Algorithms and Models <#4>`__

   -  `5.1. Principal Component Analysis <#4.1>`__
   -  `5.2. Singular Value Decomposition-SVD <#4.2>`__
   -  `5.3. Kernel Principal Component Analysis <#4.3>`__
   -  `5.4. t-SNE <#4.4>`__

 # 1. Introduction

We will look at the following models and the related concepts 1.
Principal Component Analysis (PCA) 2. Kernel PCA (KPCA) 3. t-distributed
Stochastic Neighbor Embedding (t-SNE)

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

    #Import Model Packages
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from numpy.linalg import inv, eig, svd

    from sklearn.manifold import TSNE
    from sklearn.decomposition import KernelPCA

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

    # types
    set_option('display.max_rows', 500)
    dataset.dtypes




.. parsed-literal::

    MMM     float64
    AXP     float64
    AAPL    float64
    BA      float64
    CAT     float64
    CVX     float64
    CSCO    float64
    KO      float64
    DIS     float64
    DWDP    float64
    XOM     float64
    GS      float64
    HD      float64
    IBM     float64
    INTC    float64
    JNJ     float64
    JPM     float64
    MCD     float64
    MRK     float64
    MSFT    float64
    NKE     float64
    PFE     float64
    PG      float64
    TRV     float64
    UTX     float64
    UNH     float64
    VZ      float64
    V       float64
    WMT     float64
    WBA     float64
    dtype: object



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

    <matplotlib.axes._subplots.AxesSubplot at 0x1ecdd63bdd8>




.. image:: output_20_1.png


 ## 4. Data Preparation

 ## 4.1. Data Cleaning Check for the NAs in the rows, either drop them
or fill them with the mean of the column

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

All the variables should be on the same scale before applying PCA,
otherwise a feature with large values will dominate the result. Below I
use StandardScaler in scikit-learn to standardize the dataset’s features
onto unit scale (mean = 0 and variance = 1).

.. code:: ipython3

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(dataset)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(dataset),columns = dataset.columns, index = dataset.index)
    # summarize transformed data
    rescaledDataset.head(2)




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
          <td>-1.055</td>
          <td>-0.629</td>
          <td>-0.828</td>
          <td>-0.744</td>
          <td>-1.216</td>
          <td>-1.266</td>
          <td>2.120</td>
          <td>-0.754</td>
          <td>-0.698</td>
          <td>-1.493</td>
          <td>...</td>
          <td>0.280</td>
          <td>-0.906</td>
          <td>-0.523</td>
          <td>-0.897</td>
          <td>-1.034</td>
          <td>-1.253</td>
          <td>-0.931</td>
          <td>-0.387</td>
          <td>-0.201</td>
          <td>-1.002</td>
        </tr>
        <tr>
          <th>2000-01-04</th>
          <td>-1.077</td>
          <td>-0.688</td>
          <td>-0.834</td>
          <td>-0.744</td>
          <td>-1.221</td>
          <td>-1.266</td>
          <td>1.879</td>
          <td>-0.749</td>
          <td>-0.656</td>
          <td>-1.515</td>
          <td>...</td>
          <td>0.221</td>
          <td>-0.919</td>
          <td>-0.605</td>
          <td>-0.929</td>
          <td>-1.041</td>
          <td>-1.280</td>
          <td>-0.932</td>
          <td>-0.448</td>
          <td>-0.305</td>
          <td>-1.043</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 28 columns</p>
    </div>



 # 5. Evaluate Algorithms and Models

We will look at the following Models: 1. Principal Component Analysis
(PCA) 2. Kernel PCA (KPCA) 3. t-distributed Stochastic Neighbor
Embedding (t-SNE)

 ## 5.1. Principal Component Analysis (PCA)

The idea of principal component analysis (PCA) is to reduce the
dimensionality of a dataset consisting of a large number of related
variables, while retaining as much variance in the data as possible. PCA
finds a set of new variables that the original variables are just their
linear combinations. The new variables are called Principal Components
(PCs). These principal components are orthogonal: In a 3-D case, the
principal components are perpendicular to each other. X can not be
represented by Y or Y cannot be presented by Z.

The cumulative plot shows a typical ‘elbow’ pattern that can help
identify a suitable target dimensionality because it indicates that
additional components add less explanatory value.

.. code:: ipython3

    pca = PCA()
    PrincipalComponent=pca.fit_transform(rescaledDataset)

We find that the most important factor explains around 30% of the daily
return variation. The dominant factor is usually interpreted as ‘the
market’, whereas the remaining factors can be interpreted as industry or
style factors in line with our discussion in chapters 5 and 7, depending
on the results of closer inspection (see next example).

The plot on the right shows the cumulative explained variance and
indicates that around 10 factors explain 60% of the returns of this
large cross-section of stocks.

First Principal Component /Eigenvector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    PrincipalComponent[:, 0]




.. parsed-literal::

    array([-3.51727385, -3.73472763, -3.64225264, ..., 12.28734111,
           12.38998517, 12.3841529 ])



Eigenvalues
~~~~~~~~~~~

.. code:: ipython3

    pca.explained_variance_




.. parsed-literal::

    array([2.35375812e+01, 1.91769936e+00, 6.96665482e-01, 6.24378183e-01,
           4.31320654e-01, 1.95226727e-01, 1.18718582e-01, 1.04179884e-01,
           7.38085672e-02, 5.06949081e-02, 4.62548761e-02, 3.96126584e-02,
           2.55200037e-02, 2.34257762e-02, 1.75389911e-02, 1.71545445e-02,
           1.48649870e-02, 1.36552429e-02, 1.01214103e-02, 8.60288882e-03,
           7.68205199e-03, 6.15718683e-03, 5.48535222e-03, 4.77565112e-03,
           4.68816377e-03, 4.44545487e-03, 2.87404688e-03, 2.69688798e-03])



Explained Variance
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    NumEigenvalues=5
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values().plot.barh(title='Explained Variance Ratio by Top Factors',ax=axes[0]);
    pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum().plot(ylim=(0,1),ax=axes[1], title='Cumulative Explained Variance');
    # explained_variance
    pd.Series(np.cumsum(pca.explained_variance_ratio_)).to_frame('Explained Variance_Top 5').head(5).style.format('{:,.2%}'.format)




.. raw:: html

    <style  type="text/css" >
    </style><table id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2d" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Explained Variance_Top 5</th>    </tr></thead><tbody>
                    <tr>
                            <th id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2dlevel0_row0" class="row_heading level0 row0" >0</th>
                            <td id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2drow0_col0" class="data row0 col0" >84.05%</td>
                </tr>
                <tr>
                            <th id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2dlevel0_row1" class="row_heading level0 row1" >1</th>
                            <td id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2drow1_col0" class="data row1 col0" >90.89%</td>
                </tr>
                <tr>
                            <th id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2dlevel0_row2" class="row_heading level0 row2" >2</th>
                            <td id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2drow2_col0" class="data row2 col0" >93.38%</td>
                </tr>
                <tr>
                            <th id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2dlevel0_row3" class="row_heading level0 row3" >3</th>
                            <td id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2drow3_col0" class="data row3 col0" >95.61%</td>
                </tr>
                <tr>
                            <th id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2dlevel0_row4" class="row_heading level0 row4" >4</th>
                            <td id="T_5b32f07e_ceb1_11ea_b35d_8286472efe2drow4_col0" class="data row4 col0" >97.15%</td>
                </tr>
        </tbody></table>




.. image:: output_42_1.png


Factor Loading
~~~~~~~~~~~~~~

Eigenvectors are unit-scaled loadings; and they are the coefficients
(the cosines) of orthogonal transformation (rotation) of variables into
principal components or back. Therefore it is easy to compute the
components’ values (not standardized) with them. Besides that their
usage is limited. Eigenvector value squared has the meaning of the
contribution of a variable into a pr. component; if it is high (close to
1) the component is well defined by that variable alone.

Here orthonormal eigen vectors (i.e. the term Orthonormal Eigenvectors )
provides a direction and the term Square root of (Absolute Eigen values)
provide the value.

Although eigenvectors and loadings are simply two different ways to
normalize coordinates of the same points representing columns
(variables) of the data on a biplot, it is not a good idea to mix the
two terms.

.. code:: ipython3

    loadings= (pca.components_.T*np.sqrt(pca.explained_variance_)).T

Factor loadings of the First two components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    NumComponents=2
    topComponents = pd.DataFrame(loadings[:NumComponents], columns=rescaledDataset.columns)
    eigen_Components = topComponents.div(topComponents.sum(1), axis=0)
    eigen_Components.index = [f'Principal Component {i}' for i in range(1, NumComponents+1)]
    np.sqrt(pca.explained_variance_)
    eigen_Components.T.plot.bar(subplots=True, layout=(int(NumComponents),1), figsize=(14,10), legend=False, sharey=True);



.. image:: output_47_0.png


.. code:: ipython3

    # plotting heatmap
    sns.heatmap(topComponents)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1ec82ec52e8>




.. image:: output_48_1.png


PCA to Reduce Dimension
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    pca2 = PCA(n_components=2)
    projected_data  = pca2.fit_transform(rescaledDataset)
    projected_data.shape




.. parsed-literal::

    (4804, 2)



 ## 5.2. Singular Value Decomposition (SVD)

This transformer performs linear dimensionality reduction by means of
truncated singular value decomposition (SVD). Contrary to PCA, this
estimator does not center the data before computing the singular value
decomposition.

We are using the TruncatedSVD method in the scikit-learn package
(Truncated-SVD is a quicker calculation, and using scikit-learn is more
convenient and consistent with our usage elsewhere) to transform the
full dataset into a representation using the top 300 components, thus
preserving variance in the data but using fewer dimensions/features to
do so. This has a similar effect to Principal Component Analysis (PCA)
where we represent the original data using an orthogonal set of axes
rotated and aligned to the variance in the dataset.

.. code:: ipython3

    ncomps = 20
    svd = TruncatedSVD(ncomps)
    svd_fit = svd.fit(rescaledDataset)
    Y = svd.fit_transform(rescaledDataset)

.. code:: ipython3

    NumEigenvalues=5
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    pd.Series(svd_fit.explained_variance_ratio_[:NumEigenvalues]).sort_values().plot.barh(title='Explained Variance Ratio by Top Factors',ax=axes[0]);
    pd.Series(svd_fit.explained_variance_ratio_[:NumEigenvalues]).cumsum().plot(ylim=(0,1),ax=axes[1], title='Cumulative Explained Variance');
    # explained_variance
    pd.Series(np.cumsum(svd_fit.explained_variance_ratio_)).to_frame('Explained Variance_Top 5').head(5).style.format('{:,.2%}'.format)




.. raw:: html

    <style  type="text/css" >
    </style><table id="T_5bb14654_ceb1_11ea_abab_8286472efe2d" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Explained Variance_Top 5</th>    </tr></thead><tbody>
                    <tr>
                            <th id="T_5bb14654_ceb1_11ea_abab_8286472efe2dlevel0_row0" class="row_heading level0 row0" >0</th>
                            <td id="T_5bb14654_ceb1_11ea_abab_8286472efe2drow0_col0" class="data row0 col0" >84.05%</td>
                </tr>
                <tr>
                            <th id="T_5bb14654_ceb1_11ea_abab_8286472efe2dlevel0_row1" class="row_heading level0 row1" >1</th>
                            <td id="T_5bb14654_ceb1_11ea_abab_8286472efe2drow1_col0" class="data row1 col0" >90.89%</td>
                </tr>
                <tr>
                            <th id="T_5bb14654_ceb1_11ea_abab_8286472efe2dlevel0_row2" class="row_heading level0 row2" >2</th>
                            <td id="T_5bb14654_ceb1_11ea_abab_8286472efe2drow2_col0" class="data row2 col0" >93.38%</td>
                </tr>
                <tr>
                            <th id="T_5bb14654_ceb1_11ea_abab_8286472efe2dlevel0_row3" class="row_heading level0 row3" >3</th>
                            <td id="T_5bb14654_ceb1_11ea_abab_8286472efe2drow3_col0" class="data row3 col0" >95.61%</td>
                </tr>
                <tr>
                            <th id="T_5bb14654_ceb1_11ea_abab_8286472efe2dlevel0_row4" class="row_heading level0 row4" >4</th>
                            <td id="T_5bb14654_ceb1_11ea_abab_8286472efe2drow4_col0" class="data row4 col0" >97.15%</td>
                </tr>
        </tbody></table>




.. image:: output_54_1.png


 ## 5.3. Kernel PCA (KPCA) PCA applies linear transformation, which is
just its limitation. Kernel PCA extends PCA to non-linearity. It first
maps the original data to some nonlinear feature space (usually higher
dimension), then applies PCA to extract the principal components in that
space. But if all the dots are projected onto a 3D space, the result
becomes linearly separable! We then apply PCA to separate the
components.

.. code:: ipython3

    kpca = KernelPCA(n_components=4, kernel='rbf', gamma=15)
    kpca_transform = kpca.fit_transform(rescaledDataset)
    explained_variance = np.var(kpca_transform, axis=0)
    ev = explained_variance / np.sum(explained_variance)

.. code:: ipython3

    NumEigenvalues=10
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    pd.Series(ev[:NumEigenvalues]).sort_values().plot.barh(title='Explained Variance Ratio by Top Factors',ax=axes[0]);
    pd.Series(ev[:NumEigenvalues]).cumsum().plot(ylim=(0,1),ax=axes[1], title='Cumulative Explained Variance');
    # explained_variance
    pd.Series(ev).to_frame('Explained Variance_Top 5').head(5).style.format('{:,.2%}'.format)




.. raw:: html

    <style  type="text/css" >
    </style><table id="T_5d307be8_ceb1_11ea_8c28_8286472efe2d" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Explained Variance_Top 5</th>    </tr></thead><tbody>
                    <tr>
                            <th id="T_5d307be8_ceb1_11ea_8c28_8286472efe2dlevel0_row0" class="row_heading level0 row0" >0</th>
                            <td id="T_5d307be8_ceb1_11ea_8c28_8286472efe2drow0_col0" class="data row0 col0" >26.41%</td>
                </tr>
                <tr>
                            <th id="T_5d307be8_ceb1_11ea_8c28_8286472efe2dlevel0_row1" class="row_heading level0 row1" >1</th>
                            <td id="T_5d307be8_ceb1_11ea_8c28_8286472efe2drow1_col0" class="data row1 col0" >25.96%</td>
                </tr>
                <tr>
                            <th id="T_5d307be8_ceb1_11ea_8c28_8286472efe2dlevel0_row2" class="row_heading level0 row2" >2</th>
                            <td id="T_5d307be8_ceb1_11ea_8c28_8286472efe2drow2_col0" class="data row2 col0" >24.99%</td>
                </tr>
                <tr>
                            <th id="T_5d307be8_ceb1_11ea_8c28_8286472efe2dlevel0_row3" class="row_heading level0 row3" >3</th>
                            <td id="T_5d307be8_ceb1_11ea_8c28_8286472efe2drow3_col0" class="data row3 col0" >22.64%</td>
                </tr>
        </tbody></table>




.. image:: output_57_1.png


 ## 5.4. t-distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE models the similarities among points. How does it define
similarities? First, it is defined by the Euclidean distance between
point Xi and Xj. Second, it is defined as the conditional probability
that “the similarity of data point i to point j is the conditional
probability p that point i would pick data j as its neighbor if other
neighbors were picked according to their probabilities under a Gaussian
distribution.” In the following conditional expression, if point j is
closer to point i than other points, it has a higher probability (notice
the negative sign) to be chosen.

.. code:: ipython3

    #t-SNE
    X_tsne = TSNE(learning_rate=100).fit_transform(rescaledDataset)
    X_pca = PCA().fit_transform(rescaledDataset)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x1ec81239240>




.. image:: output_59_1.png


.. code:: ipython3

    dfsvd = pd.DataFrame(Y, columns=['c{}'.format(c) for c in range(ncomps)], index=dataset.index)
    svdcols = [c for c in dfsvd.columns if c[0] == 'c']

.. code:: ipython3

    dftsne = pd.DataFrame(X_tsne, columns=['x','y'], index=dfsvd.index)

    ax = sns.lmplot('x', 'y', dftsne, fit_reg=False, size=8
                    ,scatter_kws={'alpha':0.7,'s':60})



.. image:: output_61_0.png


Pairs-plots are a simple representation using a set of 2D scatterplots,
plotting each component against another component, and coloring the
datapoints according to their classification

.. code:: ipython3

    plotdims = 5
    ploteorows = 1
    dfsvdplot = dfsvd[svdcols].iloc[:,:plotdims]
    #dfsvdplot['TYPEHUQ']=df['TYPEHUQ']
    ax = sns.pairplot(dfsvdplot.iloc[::ploteorows,:], size=1.8)



.. image:: output_63_0.png
