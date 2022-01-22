.. _AlgoTrd_BitcoinUnsup:

Dimensionality Reduction-Bitcoin Price Prediction Problem
=========================================================

In this case study, we will use the dimensionality reduction approach to
enhance the “bitcoin trading strategy” related case study discussed in
Chapter 6.

Content
-------

-  `1. Problem Definition <#0>`__
-  `2. Getting Started - Load Libraries and Dataset <#1>`__

   -  `2.1. Load Libraries <#1.1>`__
   -  `2.2. Load Dataset <#1.2>`__

-  `3. Exploratory Data Analysis <#2>`__

   -  `3.1 Descriptive Statistics <#2.1>`__

-  `4. Data Preparation <#3>`__

   -  `4.1 Data Cleaning <#3.1>`__
   -  `4.2. Preparing classification data <#3.2>`__
   -  `4.3. Feature Engineering-Constructing Technical
      Indicators <#3.3>`__
   -  `4.4.Data Visualisation <#3.4>`__

-  `5.Evaluate Algorithms and Models <#4>`__

   -  `5.1. Train/Test Split <#4.1>`__
   -  `5.2. Singular Value Decomposition-(Feature Reduction) <#4.2>`__
   -  `5.3. t-SNE visualization <#4.3>`__
   -  `5.3. Compare Models-with and without dimensionality
      Reduction <#4.4>`__

 # 1. Problem Definition

In this case study, we will use the dimensionality reduction approach to
enhance the “bitcoin trading strategy” related case study discussed in
Chapter 6.

The data and the variables used in this case study are same as the case
study presented in the classification case study chapter. The data is
the bitcoin data for the time period of Jan 2012 to October 2017, with
minute to minute updates of OHLC (Open, High, Low, Close), Volume in BTC
and indicated currency and weighted bitcoin price

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
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier

    from mpl_toolkits.mplot3d import Axes3D

    import re
    from collections import OrderedDict
    from time import time
    import sqlite3

    from scipy.linalg import svd
    from scipy import stats
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    import warnings
    warnings.filterwarnings('ignore')

    from IPython.html.widgets import interactive, fixed

 ## 2.2. Loading the Data

.. code:: ipython3

    dataset = pd.read_csv(r'../../Chapter 6 - Sup. Learning - Classification models/CaseStudy3 - Bitcoin Trading Strategy/BitstampData.csv')

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

 # 3. Exploratory Data Analysis

 ## 3.1. Descriptive Statistics

.. code:: ipython3

    # shape
    dataset.shape




.. parsed-literal::

    (2841377, 8)



.. code:: ipython3

    # peek at data
    set_option('display.width', 100)
    dataset.tail(5)




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
          <th>Timestamp</th>
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Volume_(Currency)</th>
          <th>Weighted_Price</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2841372</th>
          <td>1496188560</td>
          <td>2190.49</td>
          <td>2190.49</td>
          <td>2181.37</td>
          <td>2181.37</td>
          <td>1.700166</td>
          <td>3723.784755</td>
          <td>2190.247337</td>
        </tr>
        <tr>
          <th>2841373</th>
          <td>1496188620</td>
          <td>2190.50</td>
          <td>2197.52</td>
          <td>2186.17</td>
          <td>2195.63</td>
          <td>6.561029</td>
          <td>14402.811961</td>
          <td>2195.206304</td>
        </tr>
        <tr>
          <th>2841374</th>
          <td>1496188680</td>
          <td>2195.62</td>
          <td>2197.52</td>
          <td>2191.52</td>
          <td>2191.83</td>
          <td>15.662847</td>
          <td>34361.023647</td>
          <td>2193.791712</td>
        </tr>
        <tr>
          <th>2841375</th>
          <td>1496188740</td>
          <td>2195.82</td>
          <td>2216.00</td>
          <td>2195.82</td>
          <td>2203.51</td>
          <td>27.090309</td>
          <td>59913.492565</td>
          <td>2211.620837</td>
        </tr>
        <tr>
          <th>2841376</th>
          <td>1496188800</td>
          <td>2201.70</td>
          <td>2209.81</td>
          <td>2196.98</td>
          <td>2208.33</td>
          <td>9.961835</td>
          <td>21972.308955</td>
          <td>2205.648801</td>
        </tr>
      </tbody>
    </table>
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
          <th>Timestamp</th>
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Volume_(Currency)</th>
          <th>Weighted_Price</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>2.841e+06</td>
          <td>1.651e+06</td>
          <td>1.651e+06</td>
          <td>1.651e+06</td>
          <td>1.651e+06</td>
          <td>1.651e+06</td>
          <td>1.651e+06</td>
          <td>1.651e+06</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>1.411e+09</td>
          <td>4.959e+02</td>
          <td>4.962e+02</td>
          <td>4.955e+02</td>
          <td>4.959e+02</td>
          <td>1.188e+01</td>
          <td>5.316e+03</td>
          <td>4.959e+02</td>
        </tr>
        <tr>
          <th>std</th>
          <td>4.938e+07</td>
          <td>3.642e+02</td>
          <td>3.645e+02</td>
          <td>3.639e+02</td>
          <td>3.643e+02</td>
          <td>4.094e+01</td>
          <td>1.998e+04</td>
          <td>3.642e+02</td>
        </tr>
        <tr>
          <th>min</th>
          <td>1.325e+09</td>
          <td>3.800e+00</td>
          <td>3.800e+00</td>
          <td>1.500e+00</td>
          <td>1.500e+00</td>
          <td>0.000e+00</td>
          <td>0.000e+00</td>
          <td>3.800e+00</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>1.368e+09</td>
          <td>2.399e+02</td>
          <td>2.400e+02</td>
          <td>2.398e+02</td>
          <td>2.399e+02</td>
          <td>3.828e-01</td>
          <td>1.240e+02</td>
          <td>2.399e+02</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>1.411e+09</td>
          <td>4.200e+02</td>
          <td>4.200e+02</td>
          <td>4.199e+02</td>
          <td>4.200e+02</td>
          <td>1.823e+00</td>
          <td>6.146e+02</td>
          <td>4.200e+02</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>1.454e+09</td>
          <td>6.410e+02</td>
          <td>6.417e+02</td>
          <td>6.402e+02</td>
          <td>6.410e+02</td>
          <td>8.028e+00</td>
          <td>3.108e+03</td>
          <td>6.410e+02</td>
        </tr>
        <tr>
          <th>max</th>
          <td>1.496e+09</td>
          <td>2.755e+03</td>
          <td>2.760e+03</td>
          <td>2.752e+03</td>
          <td>2.755e+03</td>
          <td>5.854e+03</td>
          <td>1.866e+06</td>
          <td>2.754e+03</td>
        </tr>
      </tbody>
    </table>
    </div>



 # 4. Data Preparation

 ## 4.1. Data Cleaning

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())


.. parsed-literal::

    Null Values = True


Given that there are null values, we need to clean the data by filling
the *NaNs* with the last available values.

.. code:: ipython3

    dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()

.. code:: ipython3

    dataset=dataset.drop(columns=['Timestamp'])

 ## 4.2. Preparing the data for classification

We attach a label to each movement: \* **1** if the signal is that short
term price will go up as compared to the long term. \* **0** if the
signal is that short term price will go down as compared to the long
term.

.. code:: ipython3

    # Initialize the `signals` DataFrame with the `signal` column
    #datas['PriceMove'] = 0.0

    # Create short simple moving average over the short window
    dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1, center=False).mean()

    # Create long simple moving average over the long window
    dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1, center=False).mean()

    # Create signals
    dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)

.. code:: ipython3

    dataset.tail()




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
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Volume_(Currency)</th>
          <th>Weighted_Price</th>
          <th>short_mavg</th>
          <th>long_mavg</th>
          <th>signal</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2841372</th>
          <td>2190.49</td>
          <td>2190.49</td>
          <td>2181.37</td>
          <td>2181.37</td>
          <td>1.700</td>
          <td>3723.785</td>
          <td>2190.247</td>
          <td>2179.259</td>
          <td>2189.616</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2841373</th>
          <td>2190.50</td>
          <td>2197.52</td>
          <td>2186.17</td>
          <td>2195.63</td>
          <td>6.561</td>
          <td>14402.812</td>
          <td>2195.206</td>
          <td>2181.622</td>
          <td>2189.877</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2841374</th>
          <td>2195.62</td>
          <td>2197.52</td>
          <td>2191.52</td>
          <td>2191.83</td>
          <td>15.663</td>
          <td>34361.024</td>
          <td>2193.792</td>
          <td>2183.605</td>
          <td>2189.943</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2841375</th>
          <td>2195.82</td>
          <td>2216.00</td>
          <td>2195.82</td>
          <td>2203.51</td>
          <td>27.090</td>
          <td>59913.493</td>
          <td>2211.621</td>
          <td>2187.018</td>
          <td>2190.204</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2841376</th>
          <td>2201.70</td>
          <td>2209.81</td>
          <td>2196.98</td>
          <td>2208.33</td>
          <td>9.962</td>
          <td>21972.309</td>
          <td>2205.649</td>
          <td>2190.712</td>
          <td>2190.510</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 4.3. Feature Engineering

We perform feature engineering to construct technical indicators which
will be used to make the predictions, and the output variable.

The current data of the bicoin consists of date, open, high, low, close
and volume. Using this data we calculate the following technical
indicators: \* **Moving Average** : A moving average provides an
indication of the trend of the price movement by cut down the amount of
“noise” on a price chart. \* **Stochastic Oscillator %K and %D** : A
stochastic oscillator is a momentum indicator comparing a particular
closing price of a security to a range of its prices over a certain
period of time. %K and %D are slow and fast indicators. \* **Relative
Strength Index(RSI)** :It is a momentum indicator that measures the
magnitude of recent price changes to evaluate overbought or oversold
conditions in the price of a stock or other asset. \* **Rate Of
Change(ROC)**: It is a momentum oscillator, which measures the
percentage change between the current price and the n period past price.
\* **Momentum (MOM)** : It is the rate of acceleration of a security’s
price or volume – that is, the speed at which the price is changing.

.. code:: ipython3

    #calculation of exponential moving average
    def EMA(df, n):
        EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
        return EMA
    dataset['EMA10'] = EMA(dataset, 10)
    dataset['EMA30'] = EMA(dataset, 30)
    dataset['EMA200'] = EMA(dataset, 200)
    dataset.head()

    #calculation of rate of change
    def ROC(df, n):
        M = df.diff(n - 1)
        N = df.shift(n - 1)
        ROC = pd.Series(((M / N) * 100), name = 'ROC_' + str(n))
        return ROC
    dataset['ROC10'] = ROC(dataset['Close'], 10)
    dataset['ROC30'] = ROC(dataset['Close'], 30)

    #Calculation of price momentum
    def MOM(df, n):
        MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
        return MOM
    dataset['MOM10'] = MOM(dataset['Close'], 10)
    dataset['MOM30'] = MOM(dataset['Close'], 30)

    #calculation of relative strength index
    def RSI(series, period):
     delta = series.diff().dropna()
     u = delta * 0
     d = u.copy()
     u[delta > 0] = delta[delta > 0]
     d[delta < 0] = -delta[delta < 0]
     u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
     u = u.drop(u.index[:(period-1)])
     d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
     d = d.drop(d.index[:(period-1)])
     rs = u.ewm(com=period-1, adjust=False).mean() / \
     d.ewm(com=period-1, adjust=False).mean()
     return 100 - 100 / (1 + rs)
    dataset['RSI10'] = RSI(dataset['Close'], 10)
    dataset['RSI30'] = RSI(dataset['Close'], 30)
    dataset['RSI200'] = RSI(dataset['Close'], 200)

    #calculation of stochastic osillator.

    def STOK(close, low, high, n):
     STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
     return STOK

    def STOD(close, low, high, n):
     STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
     STOD = STOK.rolling(3).mean()
     return STOD

    dataset['%K10'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 10)
    dataset['%D10'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 10)
    dataset['%K30'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 30)
    dataset['%D30'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 30)
    dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
    dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

.. code:: ipython3

    #Calculation of moving average
    def MA(df, n):
        MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        return MA
    dataset['MA21'] = MA(dataset, 10)
    dataset['MA63'] = MA(dataset, 30)
    dataset['MA252'] = MA(dataset, 200)
    dataset.tail()




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
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Volume_(Currency)</th>
          <th>Weighted_Price</th>
          <th>short_mavg</th>
          <th>long_mavg</th>
          <th>signal</th>
          <th>...</th>
          <th>RSI200</th>
          <th>%K10</th>
          <th>%D10</th>
          <th>%K30</th>
          <th>%D30</th>
          <th>%K200</th>
          <th>%D200</th>
          <th>MA21</th>
          <th>MA63</th>
          <th>MA252</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2841372</th>
          <td>2190.49</td>
          <td>2190.49</td>
          <td>2181.37</td>
          <td>2181.37</td>
          <td>1.700</td>
          <td>3723.785</td>
          <td>2190.247</td>
          <td>2179.259</td>
          <td>2189.616</td>
          <td>0.0</td>
          <td>...</td>
          <td>46.613</td>
          <td>56.447</td>
          <td>73.774</td>
          <td>47.883</td>
          <td>59.889</td>
          <td>16.012</td>
          <td>18.930</td>
          <td>2179.259</td>
          <td>2182.291</td>
          <td>2220.727</td>
        </tr>
        <tr>
          <th>2841373</th>
          <td>2190.50</td>
          <td>2197.52</td>
          <td>2186.17</td>
          <td>2195.63</td>
          <td>6.561</td>
          <td>14402.812</td>
          <td>2195.206</td>
          <td>2181.622</td>
          <td>2189.877</td>
          <td>0.0</td>
          <td>...</td>
          <td>47.638</td>
          <td>93.687</td>
          <td>71.712</td>
          <td>93.805</td>
          <td>65.119</td>
          <td>26.697</td>
          <td>20.096</td>
          <td>2181.622</td>
          <td>2182.292</td>
          <td>2220.295</td>
        </tr>
        <tr>
          <th>2841374</th>
          <td>2195.62</td>
          <td>2197.52</td>
          <td>2191.52</td>
          <td>2191.83</td>
          <td>15.663</td>
          <td>34361.024</td>
          <td>2193.792</td>
          <td>2183.605</td>
          <td>2189.943</td>
          <td>0.0</td>
          <td>...</td>
          <td>47.395</td>
          <td>80.995</td>
          <td>77.043</td>
          <td>81.350</td>
          <td>74.346</td>
          <td>23.850</td>
          <td>22.186</td>
          <td>2183.605</td>
          <td>2182.120</td>
          <td>2219.802</td>
        </tr>
        <tr>
          <th>2841375</th>
          <td>2195.82</td>
          <td>2216.00</td>
          <td>2195.82</td>
          <td>2203.51</td>
          <td>27.090</td>
          <td>59913.493</td>
          <td>2211.621</td>
          <td>2187.018</td>
          <td>2190.204</td>
          <td>0.0</td>
          <td>...</td>
          <td>48.213</td>
          <td>74.205</td>
          <td>82.963</td>
          <td>74.505</td>
          <td>83.220</td>
          <td>32.602</td>
          <td>27.716</td>
          <td>2187.018</td>
          <td>2182.337</td>
          <td>2219.396</td>
        </tr>
        <tr>
          <th>2841376</th>
          <td>2201.70</td>
          <td>2209.81</td>
          <td>2196.98</td>
          <td>2208.33</td>
          <td>9.962</td>
          <td>21972.309</td>
          <td>2205.649</td>
          <td>2190.712</td>
          <td>2190.510</td>
          <td>1.0</td>
          <td>...</td>
          <td>48.545</td>
          <td>82.810</td>
          <td>79.337</td>
          <td>84.344</td>
          <td>80.066</td>
          <td>36.440</td>
          <td>30.964</td>
          <td>2190.712</td>
          <td>2182.715</td>
          <td>2218.980</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 29 columns</p>
    </div>



.. code:: ipython3

    dataset.tail()




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
          <th>Open</th>
          <th>High</th>
          <th>Low</th>
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Volume_(Currency)</th>
          <th>Weighted_Price</th>
          <th>short_mavg</th>
          <th>long_mavg</th>
          <th>signal</th>
          <th>...</th>
          <th>RSI200</th>
          <th>%K10</th>
          <th>%D10</th>
          <th>%K30</th>
          <th>%D30</th>
          <th>%K200</th>
          <th>%D200</th>
          <th>MA21</th>
          <th>MA63</th>
          <th>MA252</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2841372</th>
          <td>2190.49</td>
          <td>2190.49</td>
          <td>2181.37</td>
          <td>2181.37</td>
          <td>1.700</td>
          <td>3723.785</td>
          <td>2190.247</td>
          <td>2179.259</td>
          <td>2189.616</td>
          <td>0.0</td>
          <td>...</td>
          <td>46.613</td>
          <td>56.447</td>
          <td>73.774</td>
          <td>47.883</td>
          <td>59.889</td>
          <td>16.012</td>
          <td>18.930</td>
          <td>2179.259</td>
          <td>2182.291</td>
          <td>2220.727</td>
        </tr>
        <tr>
          <th>2841373</th>
          <td>2190.50</td>
          <td>2197.52</td>
          <td>2186.17</td>
          <td>2195.63</td>
          <td>6.561</td>
          <td>14402.812</td>
          <td>2195.206</td>
          <td>2181.622</td>
          <td>2189.877</td>
          <td>0.0</td>
          <td>...</td>
          <td>47.638</td>
          <td>93.687</td>
          <td>71.712</td>
          <td>93.805</td>
          <td>65.119</td>
          <td>26.697</td>
          <td>20.096</td>
          <td>2181.622</td>
          <td>2182.292</td>
          <td>2220.295</td>
        </tr>
        <tr>
          <th>2841374</th>
          <td>2195.62</td>
          <td>2197.52</td>
          <td>2191.52</td>
          <td>2191.83</td>
          <td>15.663</td>
          <td>34361.024</td>
          <td>2193.792</td>
          <td>2183.605</td>
          <td>2189.943</td>
          <td>0.0</td>
          <td>...</td>
          <td>47.395</td>
          <td>80.995</td>
          <td>77.043</td>
          <td>81.350</td>
          <td>74.346</td>
          <td>23.850</td>
          <td>22.186</td>
          <td>2183.605</td>
          <td>2182.120</td>
          <td>2219.802</td>
        </tr>
        <tr>
          <th>2841375</th>
          <td>2195.82</td>
          <td>2216.00</td>
          <td>2195.82</td>
          <td>2203.51</td>
          <td>27.090</td>
          <td>59913.493</td>
          <td>2211.621</td>
          <td>2187.018</td>
          <td>2190.204</td>
          <td>0.0</td>
          <td>...</td>
          <td>48.213</td>
          <td>74.205</td>
          <td>82.963</td>
          <td>74.505</td>
          <td>83.220</td>
          <td>32.602</td>
          <td>27.716</td>
          <td>2187.018</td>
          <td>2182.337</td>
          <td>2219.396</td>
        </tr>
        <tr>
          <th>2841376</th>
          <td>2201.70</td>
          <td>2209.81</td>
          <td>2196.98</td>
          <td>2208.33</td>
          <td>9.962</td>
          <td>21972.309</td>
          <td>2205.649</td>
          <td>2190.712</td>
          <td>2190.510</td>
          <td>1.0</td>
          <td>...</td>
          <td>48.545</td>
          <td>82.810</td>
          <td>79.337</td>
          <td>84.344</td>
          <td>80.066</td>
          <td>36.440</td>
          <td>30.964</td>
          <td>2190.712</td>
          <td>2182.715</td>
          <td>2218.980</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 29 columns</p>
    </div>



.. code:: ipython3

    #excluding columns that are not needed for our prediction.

    dataset=dataset.drop(['High','Low','Open', 'Volume_(Currency)','short_mavg','long_mavg'], axis=1)

.. code:: ipython3

    dataset = dataset.dropna(axis=0)

.. code:: ipython3

    dataset.tail()




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
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Weighted_Price</th>
          <th>signal</th>
          <th>EMA10</th>
          <th>EMA30</th>
          <th>EMA200</th>
          <th>ROC10</th>
          <th>ROC30</th>
          <th>MOM10</th>
          <th>...</th>
          <th>RSI200</th>
          <th>%K10</th>
          <th>%D10</th>
          <th>%K30</th>
          <th>%D30</th>
          <th>%K200</th>
          <th>%D200</th>
          <th>MA21</th>
          <th>MA63</th>
          <th>MA252</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2841372</th>
          <td>2181.37</td>
          <td>1.700</td>
          <td>2190.247</td>
          <td>0.0</td>
          <td>2181.181</td>
          <td>2182.376</td>
          <td>2211.244</td>
          <td>0.431</td>
          <td>-0.649</td>
          <td>8.42</td>
          <td>...</td>
          <td>46.613</td>
          <td>56.447</td>
          <td>73.774</td>
          <td>47.883</td>
          <td>59.889</td>
          <td>16.012</td>
          <td>18.930</td>
          <td>2179.259</td>
          <td>2182.291</td>
          <td>2220.727</td>
        </tr>
        <tr>
          <th>2841373</th>
          <td>2195.63</td>
          <td>6.561</td>
          <td>2195.206</td>
          <td>0.0</td>
          <td>2183.808</td>
          <td>2183.231</td>
          <td>2211.088</td>
          <td>1.088</td>
          <td>-0.062</td>
          <td>23.63</td>
          <td>...</td>
          <td>47.638</td>
          <td>93.687</td>
          <td>71.712</td>
          <td>93.805</td>
          <td>65.119</td>
          <td>26.697</td>
          <td>20.096</td>
          <td>2181.622</td>
          <td>2182.292</td>
          <td>2220.295</td>
        </tr>
        <tr>
          <th>2841374</th>
          <td>2191.83</td>
          <td>15.663</td>
          <td>2193.792</td>
          <td>0.0</td>
          <td>2185.266</td>
          <td>2183.786</td>
          <td>2210.897</td>
          <td>1.035</td>
          <td>-0.235</td>
          <td>19.83</td>
          <td>...</td>
          <td>47.395</td>
          <td>80.995</td>
          <td>77.043</td>
          <td>81.350</td>
          <td>74.346</td>
          <td>23.850</td>
          <td>22.186</td>
          <td>2183.605</td>
          <td>2182.120</td>
          <td>2219.802</td>
        </tr>
        <tr>
          <th>2841375</th>
          <td>2203.51</td>
          <td>27.090</td>
          <td>2211.621</td>
          <td>0.0</td>
          <td>2188.583</td>
          <td>2185.058</td>
          <td>2210.823</td>
          <td>1.479</td>
          <td>0.297</td>
          <td>34.13</td>
          <td>...</td>
          <td>48.213</td>
          <td>74.205</td>
          <td>82.963</td>
          <td>74.505</td>
          <td>83.220</td>
          <td>32.602</td>
          <td>27.716</td>
          <td>2187.018</td>
          <td>2182.337</td>
          <td>2219.396</td>
        </tr>
        <tr>
          <th>2841376</th>
          <td>2208.33</td>
          <td>9.962</td>
          <td>2205.649</td>
          <td>1.0</td>
          <td>2192.174</td>
          <td>2186.560</td>
          <td>2210.798</td>
          <td>1.626</td>
          <td>0.516</td>
          <td>36.94</td>
          <td>...</td>
          <td>48.545</td>
          <td>82.810</td>
          <td>79.337</td>
          <td>84.344</td>
          <td>80.066</td>
          <td>36.440</td>
          <td>30.964</td>
          <td>2190.712</td>
          <td>2182.715</td>
          <td>2218.980</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 23 columns</p>
    </div>



 ## 4.4. Data Visualization

.. code:: ipython3

    dataset[['Weighted_Price']].plot(grid=True)
    plt.show()



.. image:: output_35_0.png


.. code:: ipython3

    fig = plt.figure()
    plot = dataset.groupby(['signal']).size().plot(kind='barh', color='red')
    plt.show()



.. image:: output_36_0.png


The predicted variable is upward 52.87% out of total data-size, meaning
that number of the buy signals was higher than that of sell signals.

 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split

We split the dataset into 80% training set and 20% test set.

.. code:: ipython3

    # split out validation dataset for the end
    subset_dataset= dataset.iloc[-10000:]
    Y= subset_dataset["signal"]
    X = subset_dataset.loc[:, dataset.columns != 'signal']
    validation_size = 0.2
    seed = 1
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=1)

Data Standardisation
~~~~~~~~~~~~~~~~~~~~

As a preprocessing step, let’s start with normalizing the feature values
so they standardised - this makes comparisons simpler and allows next
steps for Singular Value Decomposition.

.. code:: ipython3

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns, index = X_train.index)
    # summarize transformed data
    X_train.dropna(how='any', inplace=True)
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
          <th>Close</th>
          <th>Volume_(BTC)</th>
          <th>Weighted_Price</th>
          <th>EMA10</th>
          <th>EMA30</th>
          <th>EMA200</th>
          <th>ROC10</th>
          <th>ROC30</th>
          <th>MOM10</th>
          <th>MOM30</th>
          <th>...</th>
          <th>RSI200</th>
          <th>%K10</th>
          <th>%D10</th>
          <th>%K30</th>
          <th>%D30</th>
          <th>%K200</th>
          <th>%D200</th>
          <th>MA21</th>
          <th>MA63</th>
          <th>MA252</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2834071</th>
          <td>1.072</td>
          <td>-0.367</td>
          <td>1.040</td>
          <td>1.064</td>
          <td>1.077</td>
          <td>1.014</td>
          <td>0.005</td>
          <td>-0.159</td>
          <td>0.009</td>
          <td>-0.183</td>
          <td>...</td>
          <td>-0.325</td>
          <td>1.322</td>
          <td>0.427</td>
          <td>-0.205</td>
          <td>-0.412</td>
          <td>0.714</td>
          <td>0.673</td>
          <td>1.061</td>
          <td>1.086</td>
          <td>0.895</td>
        </tr>
        <tr>
          <th>2836517</th>
          <td>-1.738</td>
          <td>1.126</td>
          <td>-1.714</td>
          <td>-1.687</td>
          <td>-1.653</td>
          <td>-1.733</td>
          <td>-0.533</td>
          <td>-0.597</td>
          <td>-0.066</td>
          <td>-0.416</td>
          <td>...</td>
          <td>-0.465</td>
          <td>-1.620</td>
          <td>-0.511</td>
          <td>-1.283</td>
          <td>-0.970</td>
          <td>-0.988</td>
          <td>-0.788</td>
          <td>-1.685</td>
          <td>-1.643</td>
          <td>-1.662</td>
        </tr>
      </tbody>
    </table>
    <p>2 rows × 22 columns</p>
    </div>



 ## 5.2. Singular Value Decomposition-(Feature Reduction)

We want to reduce the dimensionality of the problem to make it more
manageable, but at the same time we want to preserve as much information
as possible.

Hence, we use a technique called singu‐ lar value decomposition (SVD),
which is one of the ways of performing PCA.Singular Value Decomposition
(SVD) is a matrix factorization commonly used in signal processing and
data compression. We are using the TruncatedSVD method in the sklearn
package.

.. code:: ipython3

    from matplotlib.ticker import MaxNLocator
    ncomps = 5
    svd = TruncatedSVD(n_components=ncomps)
    svd_fit = svd.fit(rescaledDataset)
    plt_data = pd.DataFrame(svd_fit.explained_variance_ratio_.cumsum()*100)
    plt_data.index = np.arange(1, len(plt_data) + 1)
    Y_pred = svd.fit_transform(rescaledDataset)
    ax = plt_data.plot(kind='line', figsize=(10, 4))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Eigenvalues")
    ax.set_ylabel("Percentage Explained")
    ax.legend("")
    print('Variance preserved by first 5 components == {:.2%}'.format(svd_fit.explained_variance_ratio_.cumsum()[-1]))


.. parsed-literal::

    Variance preserved by first 5 components == 92.75%



.. image:: output_47_1.png


We can preserve 92.75% variance by using just 5 components rather than
the full 25+ original features.

.. code:: ipython3

    dfsvd = pd.DataFrame(Y_pred, columns=['c{}'.format(c) for c in range(ncomps)], index=rescaledDataset.index)
    print(dfsvd.shape)
    dfsvd.head()


.. parsed-literal::

    (8000, 5)




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
          <th>c0</th>
          <th>c1</th>
          <th>c2</th>
          <th>c3</th>
          <th>c4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2834071</th>
          <td>-2.252</td>
          <td>1.920</td>
          <td>0.538</td>
          <td>-0.019</td>
          <td>-0.967</td>
        </tr>
        <tr>
          <th>2836517</th>
          <td>5.303</td>
          <td>-1.689</td>
          <td>-0.678</td>
          <td>0.473</td>
          <td>0.643</td>
        </tr>
        <tr>
          <th>2833945</th>
          <td>-2.315</td>
          <td>-0.042</td>
          <td>1.697</td>
          <td>-1.704</td>
          <td>1.672</td>
        </tr>
        <tr>
          <th>2835048</th>
          <td>-0.977</td>
          <td>0.782</td>
          <td>3.706</td>
          <td>-0.697</td>
          <td>0.057</td>
        </tr>
        <tr>
          <th>2838804</th>
          <td>2.115</td>
          <td>-1.915</td>
          <td>0.475</td>
          <td>-0.174</td>
          <td>-0.299</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 5.2.1. Basic Visualisation of Reduced Features

Lets attempt to visualise the data with the compressed dataset,
represented by the top 5 components of an SVD.

.. code:: ipython3

    svdcols = [c for c in dfsvd.columns if c[0] == 'c']

Pairs Plots
~~~~~~~~~~~

Pairs-plots are a simple representation using a set of 2D scatterplots,
plotting each component against another component, and coloring the
datapoints according to their classification (or type of signal).

.. code:: ipython3

    plotdims = 5
    ploteorows = 1
    dfsvdplot = dfsvd[svdcols].iloc[:,:plotdims]
    dfsvdplot['signal']=Y_train
    ax = sns.pairplot(dfsvdplot.iloc[::ploteorows,:], hue='signal', size=1.8)



.. image:: output_54_0.png


**Observation**:

-  In the scatter plot of each of the principal component, we can
   clearly that there is a clear segregation of the orange and blue
   dots, which means that data-points from the same type of signal tend
   to cluster together.

-  However, it’s hard to get a full appreciation of the differences and
   similarities between data points across all the components, requiring
   that the reader hold comparisons in their head while viewing

3D Scatterplot
~~~~~~~~~~~~~~

As an alternative to the pairs-plots, we could view a 3D scatterplot,
which at least lets us see more dimensions at once and possibly get an
interactive feel for the data

.. code:: ipython3

    def scatter_3D(A, elevation=30, azimuth=120):

        maxpts=1000
        fig = plt.figure(1, figsize=(9, 9))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elevation, azim=azimuth)
        ax.set_xlabel('component 0')
        ax.set_ylabel('component 1')
        ax.set_zlabel('component 2')

        # plot subset of points
        rndpts = np.sort(np.random.choice(A.shape[0], min(maxpts,A.shape[0]), replace=False))
        coloridx = np.unique(A.iloc[rndpts]['signal'], return_inverse=True)
        colors = coloridx[1] / len(coloridx[0])

        sp = ax.scatter(A.iloc[rndpts,0], A.iloc[rndpts,1], A.iloc[rndpts,2]
                   ,c=colors, cmap="jet", marker='o', alpha=0.6
                   ,s=50, linewidths=0.8, edgecolor='#BBBBBB')

        plt.show()

.. code:: ipython3

    dfsvd['signal'] = Y_train
    interactive(scatter_3D, A=fixed(dfsvd), elevation=30, azimuth=120)



.. parsed-literal::

    interactive(children=(IntSlider(value=30, description='elevation', max=90, min=-30), IntSlider(value=120, desc…


**Observation**:

The iPython Notebook interactive package lets us create an interactive
plot with controls for elevation and azimuth We can use these controls
to interactively change the view of the top 3 components and investigate
their relations. This certainly appears to be more informative than
pairs-plots.

However, we still suffer from the same major limitations of the
pairs-plots, namely that we lose a lot of the variance and have to hold
a lot in our heads when viewing.

 ## 5.3. t-SNE visualization

In this step, we implement another technique of dimensionality reduction
- t-SNE and look at the related visualization.We will use the basic
implementation available in scikit-learn

.. code:: ipython3

    tsne = TSNE(n_components=2, random_state=0)

.. code:: ipython3

    Z = tsne.fit_transform(dfsvd[svdcols])
    dftsne = pd.DataFrame(Z, columns=['x','y'], index=dfsvd.index)

.. code:: ipython3

    dftsne['signal'] = Y_train

.. code:: ipython3

    g = sns.lmplot('x', 'y', dftsne, hue='signal', fit_reg=False, size=8
                    ,scatter_kws={'alpha':0.7,'s':60})
    g.axes.flat[0].set_title('Scatterplot of a Multiple dimension dataset reduced to 2D using t-SNE')




.. parsed-literal::

    Text(0.5, 1.0, 'Scatterplot of a Multiple dimension dataset reduced to 2D using t-SNE')




.. image:: output_65_1.png


**Observation**:

This is quite interesting way of visualizing the trading signal data.
The above plot shows us that there is a good degree of clustering for
the trading signal. Although, there are some overap of the long and
short signals, but they can be distinguished quite well using the
reduced number of features.

**In Review**:

We have analyzed the bitcoin trading signal dataset in the following
steps:

-  We prepared the data by cleaning (removing character features values,
   replacing nans) and normalizing.
-  We applied transformation during the feature reduction stage.
-  We then visualized the data in the reduced dimentionality and
   ultimately applied t-SNE algorithm to reduce the data into two
   dimensions and visualize effectivly

 ## 5.4. Compare Models-with and without dimensionality Reduction

.. code:: ipython3

    # test options for classification
    scoring = 'accuracy'

 ### 5.3.1. Models

.. code:: ipython3

    import time
    start_time = time.time()

.. code:: ipython3

    # spot check the algorithms
    models =  RandomForestClassifier(n_jobs=-1)
    cv_results_XTrain= cross_val_score(models, X_train, Y_train, cv=kfold, scoring=scoring)
    print("Time Without Dimensionality Reduction--- %s seconds ---" % (time.time() - start_time))


.. parsed-literal::

    Time Without Dimensionality Reduction--- 7.781347990036011 seconds ---


.. code:: ipython3

    start_time = time.time()
    X_SVD= dfsvd[svdcols].iloc[:,:5]
    cv_results_SVD = cross_val_score(models, X_SVD, Y_train, cv=kfold, scoring=scoring)
    print("Time with Dimensionality Reduction--- %s seconds ---" % (time.time() - start_time))


.. parsed-literal::

    Time with Dimensionality Reduction--- 2.281977653503418 seconds ---


.. code:: ipython3

    print("Result without dimensionality Reduction: %f (%f)" % (cv_results_XTrain.mean(), cv_results_XTrain.std()))
    print("Result with dimensionality Reduction: %f (%f)" % (cv_results_SVD.mean(), cv_results_SVD.std()))


.. parsed-literal::

    Result without dimensionality Reduction: 0.936375 (0.010774)
    Result with dimensionality Reduction: 0.887500 (0.012698)


Looking at the model results, we do not deviate that much from the
accuracy, and the accuracy just decreases from 93.6% to 88.7%. However,
there is a 4 times improve‐ ment in the time taken, which is
significant.

**Conclusion**:

With dimensionality reduction, we achieved almost the same accuracy with
four times improvement in the time. In trading strategy development,
when the datasets are huge and the number of features is big such
improvement in time can lead to a significant improvement in the entire
process.

We demonstrated that both SVD and t-SNE provide quite interesting way of
visualizing the trading signal data, and provide a way to distinguished
long and short signals of a trading strategy with reduced number of
features.
