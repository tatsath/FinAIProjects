.. _Port_Eigen:



Portfolio Management-Eigen Portfolio
====================================

In this case study we use dimensionality reduction techniques for
portfolio management and allocation.

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

-  `5.Evaluate Algorithms and Models <#4>`__

   -  `5.1. Train Test Split <#4.1>`__
   -  `5.2. Model Evaluation- Applying Principle Component
      Analysis <#4.2>`__

      -  `5.2.1. Explained Variance using PCA <#4.2.1>`__
      -  `5.2.2. Looking at Portfolio weights <#4.2.2>`__
      -  `5.2.3. Finding the Best Eigen Portfolio <#4.2.3>`__
      -  `5.2.4. Backtesting Eigenportfolio <#4.2.4>`__

 # 1. Problem Definition

Our goal in this case study is to maximize risk-adjusted returns using
dimensionality reduction-based algorithm on a dataset of stocks to
allocate capital into different asset classes.

The dataset used for this case study is Dow Jones Industrial Average
(DJIA) index and its respective 30 stocks from year 2000 onwards. The
dataset can be downloaded from yahoo finance.

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
    dataset = read_csv('Dow_adjcloses.csv',index_col=0)

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
          <td>29.847043</td>
          <td>35.476634</td>
          <td>3.530576</td>
          <td>26.650218</td>
          <td>14.560887</td>
          <td>21.582046</td>
          <td>43.003876</td>
          <td>16.983583</td>
          <td>23.522220</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.701180</td>
          <td>16.746856</td>
          <td>32.227726</td>
          <td>20.158885</td>
          <td>21.319030</td>
          <td>5.841355</td>
          <td>22.564221</td>
          <td>NaN</td>
          <td>47.337599</td>
          <td>21.713237</td>
        </tr>
        <tr>
          <th>2000-01-04</th>
          <td>28.661131</td>
          <td>34.134275</td>
          <td>3.232839</td>
          <td>26.610431</td>
          <td>14.372251</td>
          <td>21.582046</td>
          <td>40.577200</td>
          <td>17.040950</td>
          <td>24.899860</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.445214</td>
          <td>16.121738</td>
          <td>31.596399</td>
          <td>19.890099</td>
          <td>20.445803</td>
          <td>5.766368</td>
          <td>21.833915</td>
          <td>NaN</td>
          <td>45.566248</td>
          <td>20.907354</td>
        </tr>
        <tr>
          <th>2000-01-05</th>
          <td>30.122175</td>
          <td>33.959430</td>
          <td>3.280149</td>
          <td>28.473758</td>
          <td>14.914205</td>
          <td>22.049145</td>
          <td>40.895453</td>
          <td>17.228147</td>
          <td>25.781550</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.702157</td>
          <td>16.415912</td>
          <td>31.325831</td>
          <td>20.085579</td>
          <td>20.254784</td>
          <td>5.753327</td>
          <td>22.564221</td>
          <td>NaN</td>
          <td>44.503437</td>
          <td>21.097421</td>
        </tr>
        <tr>
          <th>2000-01-06</th>
          <td>31.877325</td>
          <td>33.959430</td>
          <td>2.996290</td>
          <td>28.553331</td>
          <td>15.459153</td>
          <td>22.903343</td>
          <td>39.781569</td>
          <td>17.210031</td>
          <td>24.899860</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.677733</td>
          <td>16.972739</td>
          <td>32.438168</td>
          <td>20.122232</td>
          <td>20.998392</td>
          <td>5.964159</td>
          <td>22.449405</td>
          <td>NaN</td>
          <td>45.126952</td>
          <td>20.527220</td>
        </tr>
        <tr>
          <th>2000-01-07</th>
          <td>32.509812</td>
          <td>34.433913</td>
          <td>3.138219</td>
          <td>29.382213</td>
          <td>15.962182</td>
          <td>23.305926</td>
          <td>42.128682</td>
          <td>18.342270</td>
          <td>24.506249</td>
          <td>NaN</td>
          <td>...</td>
          <td>4.677733</td>
          <td>18.123166</td>
          <td>35.023602</td>
          <td>20.922479</td>
          <td>21.830687</td>
          <td>6.662948</td>
          <td>22.282692</td>
          <td>NaN</td>
          <td>48.535033</td>
          <td>21.051805</td>
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

    <matplotlib.axes._subplots.AxesSubplot at 0x1e1b1d9eeb8>




.. image:: output_20_1.png


As it can be seen by the chart above, there is a significant positive
correlation between the stocks.

 ## 4. Data Preparation

 ## 4.1. Data Cleaning Let us check for the NAs in the rows, either drop
them or fill them with the mean of the column

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
    dataset= dataset.dropna(axis=0)
    # Fill na with 0
    #dataset.fillna('0')

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



Computing Daily Return

.. code:: ipython3

    # Daily Log Returns (%)
    # datareturns = np.log(data / data.shift(1))

    # Daily Linear Returns (%)
    datareturns = dataset.pct_change(1)

    #Remove Outliers beyong 3 standard deviation
    datareturns= datareturns[datareturns.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)]

 ## 4.2. Data Transformation

All the variables should be on the same scale before applying PCA,
otherwise a feature with large values will dominate the result. Below we
use StandardScaler in sklearn to standardize the dataset’s features onto
unit scale (mean = 0 and variance = 1).

Standardization is a useful technique to transform attributes to a
standard Normal distribution with a mean of 0 and a standard deviation
of 1.

.. code:: ipython3

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(datareturns)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(datareturns),columns = datareturns.columns, index = datareturns.index)
    # summarize transformed data
    datareturns.dropna(how='any', inplace=True)
    rescaledDataset.dropna(how='any', inplace=True)
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
          <th>2000-01-11</th>
          <td>-1.713</td>
          <td>0.566</td>
          <td>-2.708</td>
          <td>-1.133</td>
          <td>-1.041</td>
          <td>-0.787</td>
          <td>-1.834</td>
          <td>3.569</td>
          <td>0.725</td>
          <td>0.981</td>
          <td>...</td>
          <td>-1.936</td>
          <td>3.667</td>
          <td>-0.173</td>
          <td>1.772</td>
          <td>-0.936</td>
          <td>-1.954</td>
          <td>0.076</td>
          <td>-0.836</td>
          <td>-1.375</td>
          <td>2.942</td>
        </tr>
        <tr>
          <th>2000-01-20</th>
          <td>-3.564</td>
          <td>1.077</td>
          <td>3.304</td>
          <td>-1.670</td>
          <td>-2.834</td>
          <td>-0.446</td>
          <td>0.022</td>
          <td>0.987</td>
          <td>-2.415</td>
          <td>-1.897</td>
          <td>...</td>
          <td>-0.733</td>
          <td>-1.816</td>
          <td>-1.421</td>
          <td>-2.742</td>
          <td>-0.476</td>
          <td>-1.916</td>
          <td>1.654</td>
          <td>0.241</td>
          <td>-0.987</td>
          <td>-0.036</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 28 columns</p>
    </div>



.. code:: ipython3

    # Visualizing Log Returns for the DJIA
    plt.figure(figsize=(16, 5))
    plt.title("AAPL Return")
    plt.ylabel("Return")
    rescaledDataset.AAPL.plot()
    plt.grid(True);
    plt.legend()
    plt.show()



.. image:: output_34_0.png


 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split

The portfolio is divided into train and test split to perform the
analysis regarding the best porfolio and backtesting shown later.

.. code:: ipython3

    # Dividing the dataset into training and testing sets
    percentage = int(len(rescaledDataset) * 0.8)
    X_train = rescaledDataset[:percentage]
    X_test = rescaledDataset[percentage:]

    X_train_raw = datareturns[:percentage]
    X_test_raw = datareturns[percentage:]


    stock_tickers = rescaledDataset.columns.values
    n_tickers = len(stock_tickers)

 ## 5.2. Model Evaluation- Applying Principle Component Analysis

As this step, we create a function to compute principle component
analysis from sklearn. This function computes an inversed elbow chart
that shows the amount of principle components and how many of them
explain the variance treshold.

.. code:: ipython3

    pca = PCA()
    PrincipalComponent=pca.fit(X_train)

First Principal Component /Eigenvector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    pca.components_[0]




.. parsed-literal::

    array([-0.2278224 , -0.22835766, -0.15302828, -0.18969933, -0.20200012,
           -0.17810558, -0.19508121, -0.16845303, -0.20820442, -0.19308548,
           -0.20879404, -0.20231768, -0.19939638, -0.19521427, -0.16686975,
           -0.22806024, -0.15153408, -0.169941  , -0.19367262, -0.17118841,
           -0.18993347, -0.16805969, -0.197612  , -0.22658993, -0.13821257,
           -0.16688803, -0.16897835, -0.16070821])



 ## 5.2.1.Explained Variance using PCA

.. code:: ipython3

    NumEigenvalues=10
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    Series1 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values()*100
    Series2 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum()*100
    Series1.plot.barh(ylim=(0,9), label="woohoo",title='Explained Variance Ratio by Top 10 factors',ax=axes[0]);
    Series2.plot(ylim=(0,100),xlim=(0,9),ax=axes[1], title='Cumulative Explained Variance by factor');
    # explained_variance
    pd.Series(np.cumsum(pca.explained_variance_ratio_)).to_frame('Explained Variance').head(NumEigenvalues).style.format('{:,.2%}'.format)




.. raw:: html

    <style  type="text/css" >
    </style><table id="T_96554470_bd76_11ea_bd75_8510b281ddc8" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Explained Variance</th>    </tr></thead><tbody>
                    <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row0" class="row_heading level0 row0" >0</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row0_col0" class="data row0 col0" >37.03%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row1" class="row_heading level0 row1" >1</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row1_col0" class="data row1 col0" >42.75%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row2" class="row_heading level0 row2" >2</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row2_col0" class="data row2 col0" >47.10%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row3" class="row_heading level0 row3" >3</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row3_col0" class="data row3 col0" >51.08%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row4" class="row_heading level0 row4" >4</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row4_col0" class="data row4 col0" >54.60%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row5" class="row_heading level0 row5" >5</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row5_col0" class="data row5 col0" >57.74%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row6" class="row_heading level0 row6" >6</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row6_col0" class="data row6 col0" >60.65%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row7" class="row_heading level0 row7" >7</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row7_col0" class="data row7 col0" >63.44%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row8" class="row_heading level0 row8" >8</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row8_col0" class="data row8 col0" >66.18%</td>
                </tr>
                <tr>
                            <th id="T_96554470_bd76_11ea_bd75_8510b281ddc8level0_row9" class="row_heading level0 row9" >9</th>
                            <td id="T_96554470_bd76_11ea_bd75_8510b281ddc8row9_col0" class="data row9 col0" >68.71%</td>
                </tr>
        </tbody></table>




.. image:: output_45_1.png


We find that the most important factor explains around 40% of the daily
return variation. The dominant factor is usually interpreted as ‘the
market’, depending on the results of closer inspection.

The plot on the right shows the cumulative explained variance and
indicates that around 10 factors explain 73% of the returns of this
large cross-section of stocks.

 ## 5.2.2.Looking at Portfolio weights

We compute several functions to determine the weights of each principle
component. We then visualize a scatterplot that visualizes an organized
descending plot with the respective weight of every company at the
current chosen principle component.

.. code:: ipython3

    def PCWeights():
        '''
        Principal Components (PC) weights for each 28 PCs
        '''
        weights = pd.DataFrame()

        for i in range(len(pca.components_)):
            weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])

        weights = weights.values.T
        return weights

    weights=PCWeights()

.. code:: ipython3

    weights[0]




.. parsed-literal::

    array([0.04341287, 0.04351486, 0.02916042, 0.0361483 , 0.03849228,
           0.03393904, 0.03717385, 0.03209969, 0.03967455, 0.03679355,
           0.0397869 , 0.0385528 , 0.03799613, 0.0371992 , 0.03179799,
           0.04345819, 0.02887569, 0.03238323, 0.03690543, 0.03262094,
           0.03619291, 0.03202474, 0.0376561 , 0.04317801, 0.0263372 ,
           0.03180147, 0.0321998 , 0.03062387])



.. code:: ipython3

    pca.components_[0]




.. parsed-literal::

    array([-0.2278224 , -0.22835766, -0.15302828, -0.18969933, -0.20200012,
           -0.17810558, -0.19508121, -0.16845303, -0.20820442, -0.19308548,
           -0.20879404, -0.20231768, -0.19939638, -0.19521427, -0.16686975,
           -0.22806024, -0.15153408, -0.169941  , -0.19367262, -0.17118841,
           -0.18993347, -0.16805969, -0.197612  , -0.22658993, -0.13821257,
           -0.16688803, -0.16897835, -0.16070821])



.. code:: ipython3

    weights[0]




.. parsed-literal::

    array([0.04341287, 0.04351486, 0.02916042, 0.0361483 , 0.03849228,
           0.03393904, 0.03717385, 0.03209969, 0.03967455, 0.03679355,
           0.0397869 , 0.0385528 , 0.03799613, 0.0371992 , 0.03179799,
           0.04345819, 0.02887569, 0.03238323, 0.03690543, 0.03262094,
           0.03619291, 0.03202474, 0.0376561 , 0.04317801, 0.0263372 ,
           0.03180147, 0.0321998 , 0.03062387])



.. code:: ipython3

    NumComponents=5

    topPortfolios = pd.DataFrame(pca.components_[:NumComponents], columns=dataset.columns)
    eigen_portfolios = topPortfolios.div(topPortfolios.sum(1), axis=0)
    eigen_portfolios.index = [f'Portfolio {i}' for i in range( NumComponents)]
    np.sqrt(pca.explained_variance_)
    eigen_portfolios.T.plot.bar(subplots=True, layout=(int(NumComponents),1), figsize=(14,10), legend=False, sharey=True, ylim= (-1,1))




.. parsed-literal::

    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001E1B79E3208>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001E1B7828048>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001E1B78FA320>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001E1B798D668>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001E1B7983860>]],
          dtype=object)




.. image:: output_53_1.png


.. code:: ipython3

    # plotting heatmap
    sns.heatmap(topPortfolios)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1e1b4410898>




.. image:: output_54_1.png


The heatmap and the plot above shown the contribution of different
stocks in each eigenvector.

 ## 5.2.3. Finding the Best Eigen Portfolio

In order to find the best eigen portfolios and perform backtesting in
the next step, we use the sharpe ratio, which is a performance metric
that explains the annualized returns against the annualized volatility
of each company in a portfolio. A high sharpe ratio explains higher
returns and/or lower volatility for the specified portfolio. The
annualized sharpe ratio is computed by dividing the annualized returns
against the annualized volatility. For annualized return we apply the
geometric average of all the returns in respect to the periods per year
(days of operations in the exchange in a year). Annualized volatility is
computed by taking the standard deviation of the returns and multiplying
it by the square root of the peri‐ ods per year.

.. code:: ipython3

    # Sharpe Ratio
    def sharpe_ratio(ts_returns, periods_per_year=252):
        '''
        Sharpe ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
        It calculares the annualized return, annualized volatility, and annualized sharpe ratio.

        ts_returns are  returns of a signle eigen portfolio.
        '''
        n_years = ts_returns.shape[0]/periods_per_year
        annualized_return = np.power(np.prod(1+ts_returns),(1/n_years))-1
        annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
        annualized_sharpe = annualized_return / annualized_vol

        return annualized_return, annualized_vol, annualized_sharpe

We construct a loop to compute the principle component’s weights for
each eigen portfolio, which then uses the sharpe ratio function to look
for the portfolio with the highest sharpe ratio. Once we know which
portfolio has the highest sharpe ratio, we can visualize its performance
against the DJIA Index for comparison.

.. code:: ipython3

    def optimizedPortfolio():
        n_portfolios = len(pca.components_)
        annualized_ret = np.array([0.] * n_portfolios)
        sharpe_metric = np.array([0.] * n_portfolios)
        annualized_vol = np.array([0.] * n_portfolios)
        highest_sharpe = 0
        stock_tickers = rescaledDataset.columns.values
        n_tickers = len(stock_tickers)
        pcs = pca.components_

        for i in range(n_portfolios):

            pc_w = pcs[i] / sum(pcs[i])
            eigen_prtfi = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)
            eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)
            eigen_prti_returns = np.dot(X_train_raw.loc[:, eigen_prtfi.index], pc_w)
            eigen_prti_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_train_raw.index)
            er, vol, sharpe = sharpe_ratio(eigen_prti_returns)
            annualized_ret[i] = er
            annualized_vol[i] = vol
            sharpe_metric[i] = sharpe

            sharpe_metric= np.nan_to_num(sharpe_metric)

        # find portfolio with the highest Sharpe ratio
        highest_sharpe = np.argmax(sharpe_metric)

        print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' %
              (highest_sharpe,
               annualized_ret[highest_sharpe]*100,
               annualized_vol[highest_sharpe]*100,
               sharpe_metric[highest_sharpe]))


        fig, ax = plt.subplots()
        fig.set_size_inches(12, 4)
        ax.plot(sharpe_metric, linewidth=3)
        ax.set_title('Sharpe ratio of eigen-portfolios')
        ax.set_ylabel('Sharpe ratio')
        ax.set_xlabel('Portfolios')

        results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
        results.dropna(inplace=True)
        results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
        print(results.head(20))

        plt.show()

    optimizedPortfolio()


.. parsed-literal::

    Eigen portfolio #0 with the highest Sharpe. Return 11.47%, vol = 13.31%, Sharpe = 0.86
        Return    Vol  Sharpe
    0    0.115  0.133   0.862
    7    0.096  0.693   0.138
    5    0.100  0.845   0.118
    1    0.057  0.670   0.084
    2   -0.107  0.859  -0.124
    11  -1.000  7.228  -0.138
    13  -0.399  2.070  -0.193
    25  -1.000  5.009  -0.200
    23  -1.000  4.955  -0.202
    6   -0.416  1.967  -0.212
    10  -0.158  0.738  -0.213
    3   -0.162  0.738  -0.220
    26  -1.000  4.535  -0.220
    8   -0.422  1.397  -0.302
    17  -0.998  3.277  -0.305
    24  -0.550  1.729  -0.318
    16  -0.980  3.038  -0.323
    21  -0.470  1.420  -0.331
    14  -0.886  2.571  -0.345
    27  -0.933  2.606  -0.358



.. image:: output_60_1.png


As shown from the results above, the portfolio 12 is the best portfolio
and has the maximum sharp ratio out of all the porfolio. Let us look at
the composition of this portfolio.

.. code:: ipython3

    weights = PCWeights()
    portfolio = portfolio = pd.DataFrame()

    def plotEigen(weights, plot=False, portfolio=portfolio):
        portfolio = pd.DataFrame(data ={'weights': weights.squeeze()*100}, index = stock_tickers)
        portfolio.sort_values(by=['weights'], ascending=False, inplace=True)
        if plot:
            print('Sum of weights of current eigen-portfolio: %.2f' % np.sum(portfolio))
            portfolio.plot(title='Current Eigen-Portfolio Weights',
                figsize=(12,6),
                xticks=range(0, len(stock_tickers),1),
                rot=45,
                linewidth=3
                )
            plt.show()


        return portfolio

    # Weights are stored in arrays, where 0 is the first PC's weights.
    plotEigen(weights=weights[0], plot=True)


.. parsed-literal::

    Sum of weights of current eigen-portfolio: 100.00



.. image:: output_62_1.png




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
          <th>weights</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>AXP</th>
          <td>4.351</td>
        </tr>
        <tr>
          <th>JPM</th>
          <td>4.346</td>
        </tr>
        <tr>
          <th>MMM</th>
          <td>4.341</td>
        </tr>
        <tr>
          <th>UTX</th>
          <td>4.318</td>
        </tr>
        <tr>
          <th>GS</th>
          <td>3.979</td>
        </tr>
        <tr>
          <th>DIS</th>
          <td>3.967</td>
        </tr>
        <tr>
          <th>HD</th>
          <td>3.855</td>
        </tr>
        <tr>
          <th>CAT</th>
          <td>3.849</td>
        </tr>
        <tr>
          <th>IBM</th>
          <td>3.800</td>
        </tr>
        <tr>
          <th>TRV</th>
          <td>3.766</td>
        </tr>
        <tr>
          <th>INTC</th>
          <td>3.720</td>
        </tr>
        <tr>
          <th>CSCO</th>
          <td>3.717</td>
        </tr>
        <tr>
          <th>MSFT</th>
          <td>3.691</td>
        </tr>
        <tr>
          <th>XOM</th>
          <td>3.679</td>
        </tr>
        <tr>
          <th>PFE</th>
          <td>3.619</td>
        </tr>
        <tr>
          <th>BA</th>
          <td>3.615</td>
        </tr>
        <tr>
          <th>CVX</th>
          <td>3.394</td>
        </tr>
        <tr>
          <th>NKE</th>
          <td>3.262</td>
        </tr>
        <tr>
          <th>MRK</th>
          <td>3.238</td>
        </tr>
        <tr>
          <th>WMT</th>
          <td>3.220</td>
        </tr>
        <tr>
          <th>KO</th>
          <td>3.210</td>
        </tr>
        <tr>
          <th>PG</th>
          <td>3.202</td>
        </tr>
        <tr>
          <th>VZ</th>
          <td>3.180</td>
        </tr>
        <tr>
          <th>JNJ</th>
          <td>3.180</td>
        </tr>
        <tr>
          <th>WBA</th>
          <td>3.062</td>
        </tr>
        <tr>
          <th>AAPL</th>
          <td>2.916</td>
        </tr>
        <tr>
          <th>MCD</th>
          <td>2.888</td>
        </tr>
        <tr>
          <th>UNH</th>
          <td>2.634</td>
        </tr>
      </tbody>
    </table>
    </div>



The chart shows the allocation of the best portfolio. The weights in the
chart are in percentages.

 ## 5.2.4. Backtesting Eigenportfolio

We will now try to backtest this algorithm on the test set, by looking
at few top and bottom portfolios.

.. code:: ipython3

    def Backtest(eigen):

        '''

        Plots Principle components returns against real returns.

        '''

        eigen_prtfi = pd.DataFrame(data ={'weights': eigen.squeeze()}, index = stock_tickers)
        eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)

        eigen_prti_returns = np.dot(X_test_raw.loc[:, eigen_prtfi.index], eigen)
        eigen_portfolio_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_test_raw.index)
        returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)
        print('Current Eigen-Portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (returns*100, vol*100, sharpe))
        equal_weight_return=(X_test_raw * (1/len(pca.components_))).sum(axis=1)
        df_plot = pd.DataFrame({'EigenPorfolio Return': eigen_portfolio_returns, 'Equal Weight Index': equal_weight_return}, index=X_test.index)
        np.cumprod(df_plot + 1).plot(title='Returns of the equal weighted index vs. eigen-portfolio' ,
                              figsize=(12,6), linewidth=3)
        plt.show()

    Backtest(eigen=weights[5])
    Backtest(eigen=weights[1])
    Backtest(eigen=weights[14])


.. parsed-literal::

    Current Eigen-Portfolio:
    Return = 32.76%
    Volatility = 68.64%
    Sharpe = 0.48



.. image:: output_66_1.png


.. parsed-literal::

    Current Eigen-Portfolio:
    Return = 99.80%
    Volatility = 58.34%
    Sharpe = 1.71



.. image:: output_66_3.png


.. parsed-literal::

    Current Eigen-Portfolio:
    Return = -79.42%
    Volatility = 185.30%
    Sharpe = -0.43



.. image:: output_66_5.png


As shown in charts above the eigen portfolio return of the top
portfolios outperform the equally weighted portfolio and the eigen
portfolio ranked 19 underperformed the market significantly in the test
set.

**Conclusion**

In terms of the intuition behind the eigen portfolios, we demonstrated
that the first eigen portfolio represents a systematic risk factor and
other eigen portfolio may represent sector or industry factor. We
discuss diversification benefits offered by the eigen portfolios as they
are derived using PCA and are independent.

Looking at the backtesting result, the portfolio with the best result in
the training set leads to the best result in the test set. By using PCA,
we get independent eigen portfo‐ lios with higher return and sharp ratio
as compared to market.
