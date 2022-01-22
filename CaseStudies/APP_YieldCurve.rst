.. _APP_YieldCurve:


Yield curve prediction
----------------------

The goal of this case study is to use supervised learning-based models
to predict the yield curve. This case study is inspired by the paper
“*Artificial Neural Networks in Fixed Income Markets for Yield Curve
Forecasting*” by Nunes, Gerding, McGroarty and Niranj

Content
-------

-  `1. Introduction <#0>`__
-  `2. Getting Started - Load Libraries and Dataset <#1>`__

   -  `2.1. Load Libraries <#1.1>`__
   -  `2.2. Load Dataset <#1.2>`__

-  `3. Exploratory Data Analysis <#2>`__

   -  `3.1 Descriptive Statistics <#2.1>`__
   -  `3.2. Data Visualisation <#2.2>`__

-  `4. Data Preparation and analysis <#3>`__

   -  `4.1.Feature Selection <#3.1>`__

-  `5.Evaluate Algorithms and Models <#4>`__

   -  `5.1. Train/Test Split and Evaluation metrics <#4.1>`__]]
   -  `5.3. Compare Models and Algorithms <#4.3>`__

-  `6. Model Tuning and Grid Search <#5>`__
-  `7. Finalize the Model <#6>`__

   -  `7.1. Results and comparison of Regression and MLP <#6.1>`__

 # 1. Problem Definition

In the supervised regression framework used for this case study, three
tenors (i.e. 1M, 5Y and 30Y) of the yield curve are the predicted
variable. These tenors represent short term, medium term and long-term
tenors of the yield curve.

Features
~~~~~~~~

In order to make predictions, we use the following features:

::

   1. Previous Changes in the Treasury Curve at the following tenors: 1 Month, 3 Month, 1 Year, 2 Year, 5 Year, 7 Year, 10 Year, 30 Year

   2. Changes in % of Federal Debt held by -

       a. Public,
       b. Foreign Goverments
       c. Federal Reserve

   3. The Coporate Spread on Baa rated Debt Relative to the 10 Year

 # 2. Getting Started- Loading the data and python packages

 ## 2.1. Loading the python packages

Feature Variables
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Load libraries
    import numpy as np
    import pandas as pd
    import pandas_datareader.data as web
    from matplotlib import pyplot
    from pandas.plotting import scatter_matrix
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neural_network import MLPRegressor

    #Libraries for Deep Learning Models
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    from keras.layers import LSTM
    from keras.wrappers.scikit_learn import KerasRegressor

    #Libraries for Statistical Models
    import statsmodels.api as sm

    #Libraries for Saving the Model
    from pickle import dump
    from pickle import load

    # Time series Models
    from statsmodels.tsa.arima_model import ARIMA
    #from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Error Metrics
    from sklearn.metrics import mean_squared_error

    # Feature Selection
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, f_regression

    #Plotting
    from pandas.plotting import scatter_matrix
    from statsmodels.graphics.tsaplots import plot_acf

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

 ## 2.2. Loading the Data

.. code:: ipython3

    # Get the data by webscapping using pandas datareader
    tsy_tickers = ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10', 'DGS30',
                   'TREAST', # -- U.S. Treasury securities held by the Federal Reserve ( Millions of Dollars )
                   'FYGFDPUN', # -- Federal Debt Held by the Public ( Millions of Dollars )
                   'FDHBFIN', # -- Federal Debt Held by Foreign and International Investors ( Billions of Dollars )
                   'GFDEBTN', # -- Federal Debt: Total Public Debt ( Millions of Dollars )
                   'BAA10Y', # -- Baa Corporate Bond Yield Relative to Yield on 10-Year
                  ]
    tsy_data = web.DataReader(tsy_tickers, 'fred').dropna(how='all').ffill()
    tsy_data['FDHBFIN'] = tsy_data['FDHBFIN'] * 1000
    tsy_data['GOV_PCT'] = tsy_data['TREAST'] / tsy_data['GFDEBTN']
    tsy_data['HOM_PCT'] = tsy_data['FYGFDPUN'] / tsy_data['GFDEBTN']
    tsy_data['FOR_PCT'] = tsy_data['FDHBFIN'] / tsy_data['GFDEBTN']

.. code:: ipython3

    return_period = 5
    #Y = tsy_data.loc[:, ['DGS1MO', 'DGS5', 'DGS30']].diff(return_period).shift(-return_period)
    #return_period = 5
    Y = tsy_data.loc[:, ['DGS1MO', 'DGS5', 'DGS30']].shift(-return_period)
    Y.columns = [col+'_pred' for col in Y.columns]

    #X = tsy_data.loc[:, ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10', 'DGS30', 'GOV_PCT', 'HOM_PCT', 'FOR_PCT', 'BAA10Y']].diff(return_period)
    X = tsy_data.loc[:, ['DGS1MO', 'DGS3MO', 'DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10', 'DGS30', 'GOV_PCT', 'HOM_PCT', 'FOR_PCT', 'BAA10Y']]

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.columns]
    X = dataset.loc[:, X.columns]

.. code:: ipython3

    dataset.head()




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
          <th>DGS1MO_pred</th>
          <th>DGS5_pred</th>
          <th>DGS30_pred</th>
          <th>DGS1MO</th>
          <th>DGS3MO</th>
          <th>DGS1</th>
          <th>DGS2</th>
          <th>DGS5</th>
          <th>DGS7</th>
          <th>DGS10</th>
          <th>DGS30</th>
          <th>GOV_PCT</th>
          <th>HOM_PCT</th>
          <th>FOR_PCT</th>
          <th>BAA10Y</th>
        </tr>
        <tr>
          <th>DATE</th>
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
          <th>2010-01-06</th>
          <td>0.02</td>
          <td>2.55</td>
          <td>4.71</td>
          <td>0.03</td>
          <td>0.06</td>
          <td>0.40</td>
          <td>1.01</td>
          <td>2.60</td>
          <td>3.33</td>
          <td>3.85</td>
          <td>4.70</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.49</td>
        </tr>
        <tr>
          <th>2010-01-13</th>
          <td>0.02</td>
          <td>2.38</td>
          <td>4.50</td>
          <td>0.02</td>
          <td>0.06</td>
          <td>0.37</td>
          <td>0.97</td>
          <td>2.55</td>
          <td>3.28</td>
          <td>3.80</td>
          <td>4.71</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.50</td>
        </tr>
        <tr>
          <th>2010-01-21</th>
          <td>0.01</td>
          <td>2.41</td>
          <td>4.57</td>
          <td>0.02</td>
          <td>0.06</td>
          <td>0.31</td>
          <td>0.87</td>
          <td>2.38</td>
          <td>3.09</td>
          <td>3.62</td>
          <td>4.50</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.51</td>
        </tr>
        <tr>
          <th>2010-01-28</th>
          <td>0.04</td>
          <td>2.29</td>
          <td>4.53</td>
          <td>0.01</td>
          <td>0.08</td>
          <td>0.31</td>
          <td>0.87</td>
          <td>2.41</td>
          <td>3.15</td>
          <td>3.68</td>
          <td>4.57</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.57</td>
        </tr>
        <tr>
          <th>2010-02-04</th>
          <td>0.05</td>
          <td>2.39</td>
          <td>4.69</td>
          <td>0.04</td>
          <td>0.09</td>
          <td>0.32</td>
          <td>0.80</td>
          <td>2.29</td>
          <td>3.06</td>
          <td>3.62</td>
          <td>4.53</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.62</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    dataset




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
          <th>DGS1MO_pred</th>
          <th>DGS5_pred</th>
          <th>DGS30_pred</th>
          <th>DGS1MO</th>
          <th>DGS3MO</th>
          <th>DGS1</th>
          <th>DGS2</th>
          <th>DGS5</th>
          <th>DGS7</th>
          <th>DGS10</th>
          <th>DGS30</th>
          <th>GOV_PCT</th>
          <th>HOM_PCT</th>
          <th>FOR_PCT</th>
          <th>BAA10Y</th>
        </tr>
        <tr>
          <th>DATE</th>
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
          <th>2010-01-06</th>
          <td>0.02</td>
          <td>2.55</td>
          <td>4.71</td>
          <td>0.03</td>
          <td>0.06</td>
          <td>0.40</td>
          <td>1.01</td>
          <td>2.60</td>
          <td>3.33</td>
          <td>3.85</td>
          <td>4.70</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.49</td>
        </tr>
        <tr>
          <th>2010-01-13</th>
          <td>0.02</td>
          <td>2.38</td>
          <td>4.50</td>
          <td>0.02</td>
          <td>0.06</td>
          <td>0.37</td>
          <td>0.97</td>
          <td>2.55</td>
          <td>3.28</td>
          <td>3.80</td>
          <td>4.71</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.50</td>
        </tr>
        <tr>
          <th>2010-01-21</th>
          <td>0.01</td>
          <td>2.41</td>
          <td>4.57</td>
          <td>0.02</td>
          <td>0.06</td>
          <td>0.31</td>
          <td>0.87</td>
          <td>2.38</td>
          <td>3.09</td>
          <td>3.62</td>
          <td>4.50</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.51</td>
        </tr>
        <tr>
          <th>2010-01-28</th>
          <td>0.04</td>
          <td>2.29</td>
          <td>4.53</td>
          <td>0.01</td>
          <td>0.08</td>
          <td>0.31</td>
          <td>0.87</td>
          <td>2.41</td>
          <td>3.15</td>
          <td>3.68</td>
          <td>4.57</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.57</td>
        </tr>
        <tr>
          <th>2010-02-04</th>
          <td>0.05</td>
          <td>2.39</td>
          <td>4.69</td>
          <td>0.04</td>
          <td>0.09</td>
          <td>0.32</td>
          <td>0.80</td>
          <td>2.29</td>
          <td>3.06</td>
          <td>3.62</td>
          <td>4.53</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.62</td>
        </tr>
        <tr>
          <th>2010-02-11</th>
          <td>0.06</td>
          <td>2.48</td>
          <td>4.71</td>
          <td>0.05</td>
          <td>0.11</td>
          <td>0.38</td>
          <td>0.91</td>
          <td>2.39</td>
          <td>3.15</td>
          <td>3.73</td>
          <td>4.69</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.70</td>
        </tr>
        <tr>
          <th>2010-02-19</th>
          <td>0.09</td>
          <td>2.30</td>
          <td>4.55</td>
          <td>0.06</td>
          <td>0.11</td>
          <td>0.39</td>
          <td>0.95</td>
          <td>2.48</td>
          <td>3.24</td>
          <td>3.78</td>
          <td>4.71</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.65</td>
        </tr>
        <tr>
          <th>2010-02-26</th>
          <td>0.11</td>
          <td>2.35</td>
          <td>4.64</td>
          <td>0.09</td>
          <td>0.13</td>
          <td>0.32</td>
          <td>0.81</td>
          <td>2.30</td>
          <td>3.05</td>
          <td>3.61</td>
          <td>4.55</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.62</td>
        </tr>
        <tr>
          <th>2010-03-05</th>
          <td>0.10</td>
          <td>2.42</td>
          <td>4.62</td>
          <td>0.11</td>
          <td>0.15</td>
          <td>0.38</td>
          <td>0.91</td>
          <td>2.35</td>
          <td>3.10</td>
          <td>3.69</td>
          <td>4.64</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.61</td>
        </tr>
        <tr>
          <th>2010-03-12</th>
          <td>0.13</td>
          <td>2.48</td>
          <td>4.58</td>
          <td>0.10</td>
          <td>0.15</td>
          <td>0.41</td>
          <td>0.97</td>
          <td>2.42</td>
          <td>3.15</td>
          <td>3.71</td>
          <td>4.62</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.55</td>
        </tr>
        <tr>
          <th>2010-03-19</th>
          <td>0.11</td>
          <td>2.59</td>
          <td>4.75</td>
          <td>0.13</td>
          <td>0.16</td>
          <td>0.42</td>
          <td>1.02</td>
          <td>2.48</td>
          <td>3.16</td>
          <td>3.70</td>
          <td>4.58</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.49</td>
        </tr>
        <tr>
          <th>2010-03-26</th>
          <td>0.15</td>
          <td>2.67</td>
          <td>4.81</td>
          <td>0.11</td>
          <td>0.14</td>
          <td>0.43</td>
          <td>1.04</td>
          <td>2.59</td>
          <td>3.31</td>
          <td>3.86</td>
          <td>4.75</td>
          <td>0.061</td>
          <td>0.649</td>
          <td>0.304</td>
          <td>2.49</td>
        </tr>
        <tr>
          <th>2010-04-02</th>
          <td>0.16</td>
          <td>2.65</td>
          <td>4.74</td>
          <td>0.15</td>
          <td>0.16</td>
          <td>0.46</td>
          <td>1.11</td>
          <td>2.67</td>
          <td>3.40</td>
          <td>3.96</td>
          <td>4.81</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.44</td>
        </tr>
        <tr>
          <th>2010-04-09</th>
          <td>0.15</td>
          <td>2.49</td>
          <td>4.67</td>
          <td>0.16</td>
          <td>0.16</td>
          <td>0.46</td>
          <td>1.08</td>
          <td>2.65</td>
          <td>3.34</td>
          <td>3.90</td>
          <td>4.74</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.43</td>
        </tr>
        <tr>
          <th>2010-04-16</th>
          <td>0.14</td>
          <td>2.61</td>
          <td>4.67</td>
          <td>0.15</td>
          <td>0.16</td>
          <td>0.41</td>
          <td>0.98</td>
          <td>2.49</td>
          <td>3.20</td>
          <td>3.79</td>
          <td>4.67</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.43</td>
        </tr>
        <tr>
          <th>2010-04-23</th>
          <td>0.14</td>
          <td>2.43</td>
          <td>4.53</td>
          <td>0.14</td>
          <td>0.16</td>
          <td>0.46</td>
          <td>1.10</td>
          <td>2.61</td>
          <td>3.30</td>
          <td>3.84</td>
          <td>4.67</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.37</td>
        </tr>
        <tr>
          <th>2010-04-30</th>
          <td>0.08</td>
          <td>2.17</td>
          <td>4.28</td>
          <td>0.14</td>
          <td>0.16</td>
          <td>0.41</td>
          <td>0.97</td>
          <td>2.43</td>
          <td>3.12</td>
          <td>3.69</td>
          <td>4.53</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.38</td>
        </tr>
        <tr>
          <th>2010-05-07</th>
          <td>0.15</td>
          <td>2.16</td>
          <td>4.32</td>
          <td>0.08</td>
          <td>0.13</td>
          <td>0.38</td>
          <td>0.83</td>
          <td>2.17</td>
          <td>2.87</td>
          <td>3.45</td>
          <td>4.28</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.55</td>
        </tr>
        <tr>
          <th>2010-05-14</th>
          <td>0.17</td>
          <td>2.02</td>
          <td>4.07</td>
          <td>0.15</td>
          <td>0.16</td>
          <td>0.34</td>
          <td>0.79</td>
          <td>2.16</td>
          <td>2.87</td>
          <td>3.44</td>
          <td>4.32</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.59</td>
        </tr>
        <tr>
          <th>2010-05-21</th>
          <td>0.15</td>
          <td>2.10</td>
          <td>4.22</td>
          <td>0.17</td>
          <td>0.17</td>
          <td>0.35</td>
          <td>0.76</td>
          <td>2.02</td>
          <td>2.67</td>
          <td>3.20</td>
          <td>4.07</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.76</td>
        </tr>
        <tr>
          <th>2010-05-28</th>
          <td>0.10</td>
          <td>1.95</td>
          <td>4.11</td>
          <td>0.15</td>
          <td>0.16</td>
          <td>0.34</td>
          <td>0.76</td>
          <td>2.10</td>
          <td>2.75</td>
          <td>3.31</td>
          <td>4.22</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>2.89</td>
        </tr>
        <tr>
          <th>2010-06-07</th>
          <td>0.02</td>
          <td>2.07</td>
          <td>4.20</td>
          <td>0.10</td>
          <td>0.12</td>
          <td>0.35</td>
          <td>0.74</td>
          <td>1.95</td>
          <td>2.62</td>
          <td>3.17</td>
          <td>4.11</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>3.00</td>
        </tr>
        <tr>
          <th>2010-06-14</th>
          <td>0.05</td>
          <td>2.05</td>
          <td>4.17</td>
          <td>0.02</td>
          <td>0.07</td>
          <td>0.31</td>
          <td>0.77</td>
          <td>2.07</td>
          <td>2.73</td>
          <td>3.28</td>
          <td>4.20</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>3.06</td>
        </tr>
        <tr>
          <th>2010-06-21</th>
          <td>0.07</td>
          <td>1.83</td>
          <td>4.01</td>
          <td>0.05</td>
          <td>0.12</td>
          <td>0.29</td>
          <td>0.74</td>
          <td>2.05</td>
          <td>2.72</td>
          <td>3.26</td>
          <td>4.17</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>3.02</td>
        </tr>
        <tr>
          <th>2010-06-28</th>
          <td>0.17</td>
          <td>1.76</td>
          <td>3.89</td>
          <td>0.07</td>
          <td>0.17</td>
          <td>0.30</td>
          <td>0.62</td>
          <td>1.83</td>
          <td>2.49</td>
          <td>3.05</td>
          <td>4.01</td>
          <td>0.059</td>
          <td>0.654</td>
          <td>0.308</td>
          <td>3.10</td>
        </tr>
        <tr>
          <th>2010-07-06</th>
          <td>0.16</td>
          <td>1.90</td>
          <td>4.10</td>
          <td>0.17</td>
          <td>0.17</td>
          <td>0.32</td>
          <td>0.62</td>
          <td>1.76</td>
          <td>2.40</td>
          <td>2.95</td>
          <td>3.89</td>
          <td>0.057</td>
          <td>0.666</td>
          <td>0.319</td>
          <td>3.05</td>
        </tr>
        <tr>
          <th>2010-07-13</th>
          <td>0.16</td>
          <td>1.71</td>
          <td>3.99</td>
          <td>0.16</td>
          <td>0.15</td>
          <td>0.30</td>
          <td>0.67</td>
          <td>1.90</td>
          <td>2.57</td>
          <td>3.15</td>
          <td>4.10</td>
          <td>0.057</td>
          <td>0.666</td>
          <td>0.319</td>
          <td>3.03</td>
        </tr>
        <tr>
          <th>2010-07-20</th>
          <td>0.16</td>
          <td>1.82</td>
          <td>4.08</td>
          <td>0.16</td>
          <td>0.16</td>
          <td>0.27</td>
          <td>0.61</td>
          <td>1.71</td>
          <td>2.39</td>
          <td>2.98</td>
          <td>3.99</td>
          <td>0.057</td>
          <td>0.666</td>
          <td>0.319</td>
          <td>2.99</td>
        </tr>
        <tr>
          <th>2010-07-27</th>
          <td>0.15</td>
          <td>1.55</td>
          <td>4.04</td>
          <td>0.16</td>
          <td>0.15</td>
          <td>0.30</td>
          <td>0.65</td>
          <td>1.82</td>
          <td>2.51</td>
          <td>3.08</td>
          <td>4.08</td>
          <td>0.057</td>
          <td>0.666</td>
          <td>0.319</td>
          <td>2.90</td>
        </tr>
        <tr>
          <th>2010-08-03</th>
          <td>0.15</td>
          <td>1.46</td>
          <td>4.00</td>
          <td>0.15</td>
          <td>0.16</td>
          <td>0.27</td>
          <td>0.53</td>
          <td>1.55</td>
          <td>2.27</td>
          <td>2.94</td>
          <td>4.04</td>
          <td>0.057</td>
          <td>0.666</td>
          <td>0.319</td>
          <td>2.92</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2019-05-31</th>
          <td>2.30</td>
          <td>1.85</td>
          <td>2.57</td>
          <td>2.35</td>
          <td>2.35</td>
          <td>2.21</td>
          <td>1.95</td>
          <td>1.93</td>
          <td>2.03</td>
          <td>2.14</td>
          <td>2.58</td>
          <td>0.096</td>
          <td>0.736</td>
          <td>0.302</td>
          <td>2.37</td>
        </tr>
        <tr>
          <th>2019-06-07</th>
          <td>2.23</td>
          <td>1.85</td>
          <td>2.59</td>
          <td>2.30</td>
          <td>2.28</td>
          <td>1.97</td>
          <td>1.85</td>
          <td>1.85</td>
          <td>1.97</td>
          <td>2.09</td>
          <td>2.57</td>
          <td>0.096</td>
          <td>0.736</td>
          <td>0.302</td>
          <td>2.42</td>
        </tr>
        <tr>
          <th>2019-06-14</th>
          <td>2.16</td>
          <td>1.80</td>
          <td>2.59</td>
          <td>2.23</td>
          <td>2.20</td>
          <td>2.00</td>
          <td>1.84</td>
          <td>1.85</td>
          <td>1.96</td>
          <td>2.09</td>
          <td>2.59</td>
          <td>0.096</td>
          <td>0.736</td>
          <td>0.302</td>
          <td>2.42</td>
        </tr>
        <tr>
          <th>2019-06-21</th>
          <td>2.18</td>
          <td>1.76</td>
          <td>2.52</td>
          <td>2.16</td>
          <td>2.11</td>
          <td>1.95</td>
          <td>1.77</td>
          <td>1.80</td>
          <td>1.93</td>
          <td>2.07</td>
          <td>2.59</td>
          <td>0.096</td>
          <td>0.736</td>
          <td>0.302</td>
          <td>2.35</td>
        </tr>
        <tr>
          <th>2019-06-28</th>
          <td>2.23</td>
          <td>1.86</td>
          <td>2.53</td>
          <td>2.18</td>
          <td>2.12</td>
          <td>1.92</td>
          <td>1.75</td>
          <td>1.76</td>
          <td>1.87</td>
          <td>2.00</td>
          <td>2.52</td>
          <td>0.096</td>
          <td>0.736</td>
          <td>0.302</td>
          <td>2.31</td>
        </tr>
        <tr>
          <th>2019-07-08</th>
          <td>2.17</td>
          <td>1.84</td>
          <td>2.61</td>
          <td>2.23</td>
          <td>2.26</td>
          <td>1.99</td>
          <td>1.88</td>
          <td>1.86</td>
          <td>1.94</td>
          <td>2.05</td>
          <td>2.53</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.23</td>
        </tr>
        <tr>
          <th>2019-07-15</th>
          <td>2.13</td>
          <td>1.80</td>
          <td>2.58</td>
          <td>2.17</td>
          <td>2.16</td>
          <td>1.95</td>
          <td>1.83</td>
          <td>1.84</td>
          <td>1.96</td>
          <td>2.09</td>
          <td>2.61</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.25</td>
        </tr>
        <tr>
          <th>2019-07-22</th>
          <td>2.12</td>
          <td>1.84</td>
          <td>2.59</td>
          <td>2.13</td>
          <td>2.09</td>
          <td>1.95</td>
          <td>1.80</td>
          <td>1.80</td>
          <td>1.92</td>
          <td>2.05</td>
          <td>2.58</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.24</td>
        </tr>
        <tr>
          <th>2019-07-29</th>
          <td>2.07</td>
          <td>1.55</td>
          <td>2.30</td>
          <td>2.12</td>
          <td>2.12</td>
          <td>1.98</td>
          <td>1.85</td>
          <td>1.84</td>
          <td>1.93</td>
          <td>2.06</td>
          <td>2.59</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.13</td>
        </tr>
        <tr>
          <th>2019-08-05</th>
          <td>2.09</td>
          <td>1.49</td>
          <td>2.14</td>
          <td>2.07</td>
          <td>2.05</td>
          <td>1.78</td>
          <td>1.59</td>
          <td>1.55</td>
          <td>1.63</td>
          <td>1.75</td>
          <td>2.30</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.24</td>
        </tr>
        <tr>
          <th>2019-08-12</th>
          <td>2.06</td>
          <td>1.47</td>
          <td>2.08</td>
          <td>2.09</td>
          <td>2.00</td>
          <td>1.75</td>
          <td>1.58</td>
          <td>1.49</td>
          <td>1.56</td>
          <td>1.65</td>
          <td>2.14</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.23</td>
        </tr>
        <tr>
          <th>2019-08-19</th>
          <td>2.09</td>
          <td>1.43</td>
          <td>2.04</td>
          <td>2.06</td>
          <td>1.94</td>
          <td>1.75</td>
          <td>1.53</td>
          <td>1.47</td>
          <td>1.54</td>
          <td>1.60</td>
          <td>2.08</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.29</td>
        </tr>
        <tr>
          <th>2019-08-26</th>
          <td>2.06</td>
          <td>1.35</td>
          <td>1.95</td>
          <td>2.09</td>
          <td>2.01</td>
          <td>1.75</td>
          <td>1.54</td>
          <td>1.43</td>
          <td>1.49</td>
          <td>1.54</td>
          <td>2.04</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.27</td>
        </tr>
        <tr>
          <th>2019-09-03</th>
          <td>2.04</td>
          <td>1.58</td>
          <td>2.19</td>
          <td>2.06</td>
          <td>1.98</td>
          <td>1.72</td>
          <td>1.47</td>
          <td>1.35</td>
          <td>1.42</td>
          <td>1.47</td>
          <td>1.95</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.28</td>
        </tr>
        <tr>
          <th>2019-09-10</th>
          <td>2.10</td>
          <td>1.66</td>
          <td>2.27</td>
          <td>2.04</td>
          <td>1.95</td>
          <td>1.81</td>
          <td>1.67</td>
          <td>1.58</td>
          <td>1.66</td>
          <td>1.72</td>
          <td>2.19</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.22</td>
        </tr>
        <tr>
          <th>2019-09-17</th>
          <td>1.90</td>
          <td>1.52</td>
          <td>2.09</td>
          <td>2.10</td>
          <td>1.99</td>
          <td>1.87</td>
          <td>1.72</td>
          <td>1.66</td>
          <td>1.75</td>
          <td>1.81</td>
          <td>2.27</td>
          <td>0.092</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.21</td>
        </tr>
        <tr>
          <th>2019-09-24</th>
          <td>1.79</td>
          <td>1.51</td>
          <td>2.11</td>
          <td>1.90</td>
          <td>1.92</td>
          <td>1.78</td>
          <td>1.60</td>
          <td>1.52</td>
          <td>1.58</td>
          <td>1.64</td>
          <td>2.09</td>
          <td>0.093</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.20</td>
        </tr>
        <tr>
          <th>2019-10-01</th>
          <td>1.69</td>
          <td>1.36</td>
          <td>2.04</td>
          <td>1.79</td>
          <td>1.82</td>
          <td>1.73</td>
          <td>1.56</td>
          <td>1.51</td>
          <td>1.59</td>
          <td>1.65</td>
          <td>2.11</td>
          <td>0.093</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.22</td>
        </tr>
        <tr>
          <th>2019-10-08</th>
          <td>1.71</td>
          <td>1.57</td>
          <td>2.23</td>
          <td>1.69</td>
          <td>1.72</td>
          <td>1.62</td>
          <td>1.42</td>
          <td>1.36</td>
          <td>1.45</td>
          <td>1.54</td>
          <td>2.04</td>
          <td>0.093</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.30</td>
        </tr>
        <tr>
          <th>2019-10-16</th>
          <td>1.74</td>
          <td>1.58</td>
          <td>2.25</td>
          <td>1.71</td>
          <td>1.66</td>
          <td>1.59</td>
          <td>1.58</td>
          <td>1.57</td>
          <td>1.65</td>
          <td>1.75</td>
          <td>2.23</td>
          <td>0.093</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.22</td>
        </tr>
        <tr>
          <th>2019-10-23</th>
          <td>1.61</td>
          <td>1.61</td>
          <td>2.26</td>
          <td>1.74</td>
          <td>1.65</td>
          <td>1.58</td>
          <td>1.58</td>
          <td>1.58</td>
          <td>1.67</td>
          <td>1.77</td>
          <td>2.25</td>
          <td>0.095</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.18</td>
        </tr>
        <tr>
          <th>2019-10-30</th>
          <td>1.55</td>
          <td>1.63</td>
          <td>2.30</td>
          <td>1.61</td>
          <td>1.62</td>
          <td>1.59</td>
          <td>1.61</td>
          <td>1.61</td>
          <td>1.69</td>
          <td>1.78</td>
          <td>2.26</td>
          <td>0.096</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.17</td>
        </tr>
        <tr>
          <th>2019-11-06</th>
          <td>1.59</td>
          <td>1.63</td>
          <td>2.31</td>
          <td>1.55</td>
          <td>1.56</td>
          <td>1.58</td>
          <td>1.61</td>
          <td>1.63</td>
          <td>1.73</td>
          <td>1.81</td>
          <td>2.30</td>
          <td>0.097</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.14</td>
        </tr>
        <tr>
          <th>2019-11-14</th>
          <td>1.57</td>
          <td>1.62</td>
          <td>2.24</td>
          <td>1.59</td>
          <td>1.57</td>
          <td>1.55</td>
          <td>1.58</td>
          <td>1.63</td>
          <td>1.73</td>
          <td>1.82</td>
          <td>2.31</td>
          <td>0.097</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.13</td>
        </tr>
        <tr>
          <th>2019-11-21</th>
          <td>1.62</td>
          <td>1.62</td>
          <td>2.21</td>
          <td>1.57</td>
          <td>1.58</td>
          <td>1.55</td>
          <td>1.60</td>
          <td>1.62</td>
          <td>1.71</td>
          <td>1.77</td>
          <td>2.24</td>
          <td>0.098</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.15</td>
        </tr>
        <tr>
          <th>2019-11-29</th>
          <td>1.52</td>
          <td>1.67</td>
          <td>2.29</td>
          <td>1.62</td>
          <td>1.59</td>
          <td>1.60</td>
          <td>1.61</td>
          <td>1.62</td>
          <td>1.73</td>
          <td>1.78</td>
          <td>2.21</td>
          <td>0.099</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.08</td>
        </tr>
        <tr>
          <th>2019-12-06</th>
          <td>1.55</td>
          <td>1.66</td>
          <td>2.26</td>
          <td>1.52</td>
          <td>1.53</td>
          <td>1.57</td>
          <td>1.61</td>
          <td>1.67</td>
          <td>1.78</td>
          <td>1.84</td>
          <td>2.29</td>
          <td>0.099</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.07</td>
        </tr>
        <tr>
          <th>2019-12-13</th>
          <td>1.57</td>
          <td>1.73</td>
          <td>2.34</td>
          <td>1.55</td>
          <td>1.57</td>
          <td>1.54</td>
          <td>1.61</td>
          <td>1.66</td>
          <td>1.76</td>
          <td>1.82</td>
          <td>2.26</td>
          <td>0.100</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>2.02</td>
        </tr>
        <tr>
          <th>2019-12-20</th>
          <td>1.56</td>
          <td>1.68</td>
          <td>2.32</td>
          <td>1.57</td>
          <td>1.58</td>
          <td>1.52</td>
          <td>1.63</td>
          <td>1.73</td>
          <td>1.84</td>
          <td>1.92</td>
          <td>2.34</td>
          <td>0.101</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>1.96</td>
        </tr>
        <tr>
          <th>2019-12-27</th>
          <td>1.52</td>
          <td>1.59</td>
          <td>2.26</td>
          <td>1.56</td>
          <td>1.57</td>
          <td>1.51</td>
          <td>1.59</td>
          <td>1.68</td>
          <td>1.80</td>
          <td>1.88</td>
          <td>2.32</td>
          <td>0.103</td>
          <td>0.741</td>
          <td>0.298</td>
          <td>1.96</td>
        </tr>
      </tbody>
    </table>
    <p>505 rows × 15 columns</p>
    </div>



 # 3. Exploratory Data Analysis

 ## 3.1. Descriptive Statistics

.. code:: ipython3

    dataset.shape




.. parsed-literal::

    (505, 15)



.. code:: ipython3

    pd.set_option('precision', 3)
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
          <th>DGS1MO_pred</th>
          <th>DGS5_pred</th>
          <th>DGS30_pred</th>
          <th>DGS1MO</th>
          <th>DGS3MO</th>
          <th>DGS1</th>
          <th>DGS2</th>
          <th>DGS5</th>
          <th>DGS7</th>
          <th>DGS10</th>
          <th>DGS30</th>
          <th>GOV_PCT</th>
          <th>HOM_PCT</th>
          <th>FOR_PCT</th>
          <th>BAA10Y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
          <td>505.000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>0.544</td>
          <td>1.644</td>
          <td>3.180</td>
          <td>0.541</td>
          <td>0.578</td>
          <td>0.746</td>
          <td>0.961</td>
          <td>1.646</td>
          <td>2.041</td>
          <td>2.400</td>
          <td>3.185</td>
          <td>0.110</td>
          <td>0.711</td>
          <td>0.320</td>
          <td>2.588</td>
        </tr>
        <tr>
          <th>std</th>
          <td>0.780</td>
          <td>0.593</td>
          <td>0.612</td>
          <td>0.779</td>
          <td>0.797</td>
          <td>0.810</td>
          <td>0.750</td>
          <td>0.595</td>
          <td>0.558</td>
          <td>0.553</td>
          <td>0.614</td>
          <td>0.022</td>
          <td>0.023</td>
          <td>0.018</td>
          <td>0.451</td>
        </tr>
        <tr>
          <th>min</th>
          <td>0.000</td>
          <td>0.570</td>
          <td>1.950</td>
          <td>0.000</td>
          <td>0.000</td>
          <td>0.090</td>
          <td>0.180</td>
          <td>0.570</td>
          <td>0.930</td>
          <td>1.400</td>
          <td>1.950</td>
          <td>0.057</td>
          <td>0.649</td>
          <td>0.285</td>
          <td>1.580</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>0.030</td>
          <td>1.250</td>
          <td>2.800</td>
          <td>0.030</td>
          <td>0.050</td>
          <td>0.160</td>
          <td>0.370</td>
          <td>1.250</td>
          <td>1.600</td>
          <td>1.960</td>
          <td>2.810</td>
          <td>0.101</td>
          <td>0.702</td>
          <td>0.304</td>
          <td>2.240</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>0.120</td>
          <td>1.620</td>
          <td>3.040</td>
          <td>0.110</td>
          <td>0.130</td>
          <td>0.300</td>
          <td>0.680</td>
          <td>1.620</td>
          <td>2.050</td>
          <td>2.340</td>
          <td>3.050</td>
          <td>0.113</td>
          <td>0.720</td>
          <td>0.324</td>
          <td>2.600</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>0.950</td>
          <td>1.960</td>
          <td>3.450</td>
          <td>0.850</td>
          <td>1.000</td>
          <td>1.220</td>
          <td>1.330</td>
          <td>1.970</td>
          <td>2.370</td>
          <td>2.760</td>
          <td>3.450</td>
          <td>0.126</td>
          <td>0.724</td>
          <td>0.338</td>
          <td>2.900</td>
        </tr>
        <tr>
          <th>max</th>
          <td>2.450</td>
          <td>3.070</td>
          <td>4.810</td>
          <td>2.450</td>
          <td>2.480</td>
          <td>2.740</td>
          <td>2.960</td>
          <td>3.070</td>
          <td>3.400</td>
          <td>3.960</td>
          <td>4.810</td>
          <td>0.137</td>
          <td>0.741</td>
          <td>0.341</td>
          <td>3.600</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 3.2. Data Visualization

.. code:: ipython3

    Y.plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x25e4c803b70>




.. image:: output_19_1.png


.. code:: ipython3

    # histograms
    dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
    pyplot.show()



.. image:: output_20_0.png


.. code:: ipython3

    # density
    dataset.plot(kind='density', subplots=True, layout=(5,4), sharex=False, legend=True, fontsize=1, figsize=(15,15))
    pyplot.show()



.. image:: output_21_0.png


.. code:: ipython3

    #Box and Whisker Plots
    dataset.plot(kind='box', subplots=True, layout=(5,4), sharex=False, sharey=False, figsize=(15,15))
    pyplot.show()



.. image:: output_22_0.png


Next We look at the interaction between these variables.

.. code:: ipython3

    # correlation
    correlation = dataset.corr()
    pyplot.figure(figsize=(15,15))
    pyplot.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x25e3971e358>




.. image:: output_24_1.png


Form the correlation plot, we see that the 1 month and the 30 year yield
data points are negatively autocorrelated. The 5 year yield also seems
toe be negativly correlated with the changes in foreign goverment
purchases.

.. code:: ipython3

    # Scatterplot Matrix
    pyplot.figure(figsize=(15,15))
    scatter_matrix(dataset,figsize=(15,16))
    pyplot.show()



.. parsed-literal::

    <Figure size 1080x1080 with 0 Axes>



.. image:: output_26_1.png


 ## 3.3. Time Series Analysis

1 Month
^^^^^^^

.. code:: ipython3

    temp_Y = dataset['DGS1MO_pred']
    res = sm.tsa.seasonal_decompose(temp_Y,freq=52)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()



.. image:: output_29_0.png


5 Year
^^^^^^

.. code:: ipython3

    temp_Y = dataset['DGS5_pred']
    res = sm.tsa.seasonal_decompose(temp_Y,freq=52)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()



.. image:: output_31_0.png


30 Year
^^^^^^^

.. code:: ipython3

    temp_Y = dataset['DGS30_pred']
    res = sm.tsa.seasonal_decompose(temp_Y,freq=52)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()



.. image:: output_33_0.png


Around Q1 2018, we observe a trend decrease in the 1 Month, 5 year and
30 year. However, the trend is most pronounced in the 1 month series.

 ## 4. Data Preparation and analysis

 ## 4.1. Univariate Feature Selection

.. code:: ipython3

    bestfeatures = SelectKBest(k=5, score_func=f_regression)
    for col in Y.columns:
        temp_Y = dataset[col]
        temp_X = dataset.loc[:, X.columns]
        fit = bestfeatures.fit(temp_X,temp_Y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        print(col)
        print(featureScores.nlargest(10,'Score'))  #print 10 best features
        print('--------------')


.. parsed-literal::

    DGS1MO_pred
          Specs       Score
    0    DGS1MO  152945.490
    1    DGS3MO  100807.006
    2      DGS1   11168.249
    3      DGS2    3510.465
    10  FOR_PCT    1010.264
    11   BAA10Y     361.423
    4      DGS5     342.936
    9   HOM_PCT     243.490
    5      DGS7      85.197
    7     DGS30      59.793
    --------------
    DGS5_pred
          Specs      Score
    4      DGS5  16564.639
    5      DGS7   2720.883
    3      DGS2    970.840
    10  FOR_PCT    613.935
    11   BAA10Y    586.571
    6     DGS10    517.453
    2      DGS1    505.191
    1    DGS3MO    363.006
    0    DGS1MO    326.866
    7     DGS30     39.443
    --------------
    DGS30_pred
         Specs      Score
    7    DGS30  17108.682
    6    DGS10   1235.241
    9  HOM_PCT    629.480
    5     DGS7    252.017
    8  GOV_PCT    183.786
    0   DGS1MO     62.107
    1   DGS3MO     60.412
    2     DGS1     55.316
    4     DGS5     40.429
    3     DGS2     25.822
    --------------


As expected, based on the univariate feature selection, all the time
series are most dependent on the previous changes.

 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split and evaluation metrics

.. code:: ipython3

    # split out validation dataset for the end

    validation_size = 0.2

    #In case the data is not dependent on the time series, then train and test split randomly
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

We use the prebuilt scikit models to run a K fold analysis on our
training data. We then train the model on the full training data and use
it for prediction of the test data. The parameters for the K fold
analysis are defined as -

.. code:: ipython3

    # test options for regression
    num_folds = 10
    scoring = 'neg_mean_squared_error'

 ## 5.2. Compare Models and Algorithms

.. code:: ipython3

    # spot check the algorithms
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    #Neural Network
    models.append(('MLP', MLPRegressor()))

.. code:: ipython3

    kfold_results = []
    names = []
    validation_results = []
    train_results = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        #converted mean square error to positive. The lower the beter
        cv_results = -1* cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)
        names.append(name)

        # Finally we Train on the full period and test against validation
        res = model.fit(X_train, Y_train)
        validation_result = np.mean(np.square(res.predict(X_validation) - Y_validation))
        validation_results.append(validation_result)
        train_result = np.mean(np.square(res.predict(X_train) - Y_train))
        train_results.append(train_result)

        msg = "%s: \nAverage CV error: %s \nStd CV Error: (%s) \nTraining Error:\n%s \nTest Error:\n%s" % \
        (name, str(cv_results.mean()), str(cv_results.std()), str(train_result), str(validation_result))
        print(msg)
        print('----------')


.. parsed-literal::

    LR:
    Average CV error: 0.007942891864184351
    Std CV Error: (0.002627539557566172)
    Training Error:
    DGS1MO_pred    0.002
    DGS5_pred      0.010
    DGS30_pred     0.010
    dtype: float64
    Test Error:
    DGS1MO_pred    0.001
    DGS5_pred      0.009
    DGS30_pred     0.010
    dtype: float64
    ----------
    LASSO:
    Average CV error: 0.44035581436209686
    Std CV Error: (0.05366688435468398)
    Training Error:
    DGS1MO_pred    0.574
    DGS5_pred      0.352
    DGS30_pred     0.388
    dtype: float64
    Test Error:
    DGS1MO_pred    0.743
    DGS5_pred      0.350
    DGS30_pred     0.318
    dtype: float64
    ----------
    EN:
    Average CV error: 0.4019832745740823
    Std CV Error: (0.050762635401215755)
    Training Error:
    DGS1MO_pred    0.455
    DGS5_pred      0.352
    DGS30_pred     0.388
    dtype: float64
    Test Error:
    DGS1MO_pred    0.592
    DGS5_pred      0.350
    DGS30_pred     0.318
    dtype: float64
    ----------
    KNN:
    Average CV error: 0.009184607723577237
    Std CV Error: (0.003118097916218511)
    Training Error:
    DGS1MO_pred    0.002
    DGS5_pred      0.008
    DGS30_pred     0.007
    dtype: float64
    Test Error:
    DGS1MO_pred    0.002
    DGS5_pred      0.011
    DGS30_pred     0.011
    dtype: float64
    ----------
    CART:
    Average CV error: 0.017137121951219508
    Std CV Error: (0.003920838626872242)
    Training Error:
    DGS1MO_pred    0.0
    DGS5_pred      0.0
    DGS30_pred     0.0
    dtype: float64
    Test Error:
    DGS1MO_pred    0.003
    DGS5_pred      0.019
    DGS30_pred     0.017
    dtype: float64
    ----------
    MLP:
    Average CV error: 0.015686185139476356
    Std CV Error: (0.004980051035156162)
    Training Error:
    DGS1MO_pred    0.003
    DGS5_pred      0.011
    DGS30_pred     0.025
    dtype: float64
    Test Error:
    DGS1MO_pred    0.002
    DGS5_pred      0.009
    DGS30_pred     0.026
    dtype: float64
    ----------


.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(kfold_results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15,8)
    pyplot.show()



.. image:: output_47_0.png


.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()

    ind = np.arange(len(names))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.bar(ind - width/2, [x.mean() for x in train_results],  width=width, label='Train Error')
    pyplot.bar(ind + width/2, [x.mean() for x in validation_results], width=width, label='Validation Error')
    fig.set_size_inches(15,8)
    pyplot.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    pyplot.show()



.. image:: output_48_0.png


 # 6. Model Tuning and Grid Search

.. code:: ipython3

    # 7. Grid search : MLPRegressor
    '''
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    '''
    param_grid={'hidden_layer_sizes': [(20,), (50,), (20,20), (20, 30, 20)]}
    model = MLPRegressor()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


.. parsed-literal::

    Best: -0.018006 using {'hidden_layer_sizes': (20, 30, 20)}
    -0.036433 (0.019326) with: {'hidden_layer_sizes': (20,)}
    -0.020793 (0.007075) with: {'hidden_layer_sizes': (50,)}
    -0.026638 (0.010154) with: {'hidden_layer_sizes': (20, 20)}
    -0.018006 (0.005637) with: {'hidden_layer_sizes': (20, 30, 20)}


 # 7. Finalise the Model

.. code:: ipython3

    # prepare model
    model = MLPRegressor(hidden_layer_sizes= (20, 30, 20))
    model.fit(X_train, Y_train)




.. parsed-literal::

    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                 beta_2=0.999, early_stopping=False, epsilon=1e-08,
                 hidden_layer_sizes=(20, 30, 20), learning_rate='constant',
                 learning_rate_init=0.001, max_iter=200, momentum=0.9,
                 n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                 random_state=None, shuffle=True, solver='adam', tol=0.0001,
                 validation_fraction=0.1, verbose=False, warm_start=False)



 ## 7.1. Results and comparison of Regression and MLP

.. code:: ipython3

    # estimate accuracy on validation set
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    predictions = model.predict(X_validation)
    mse_MLP = mean_squared_error(Y_validation, predictions)
    r2_MLP = r2_score(Y_validation, predictions)

    # prepare model
    model_2 = LinearRegression()
    model_2.fit(X_train, Y_train)
    predictions_2 = model_2.predict(X_validation)

    mse_OLS = mean_squared_error(Y_validation, predictions_2)
    r2_OLS = r2_score(Y_validation, predictions_2)
    print("MSE Regression = %f, MSE MLP = %f" % (mse_OLS, mse_MLP ))
    print("R2 Regression = %f, R2 MLP = %f" % (r2_OLS, r2_MLP ))



.. parsed-literal::

    MSE Regression = 0.006727, MSE MLP = 0.015661
    R2 Regression = 0.979949, R2 MLP = 0.954716


The statistics of MLP and Linear regression are comparable. Let us check
the prediction shape on the validation set.

Predictions - 5 Year - MLP
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    train_size = int(len(X) * (1-validation_size))
    X_train, X_validation = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_validation = Y[0:train_size], Y[train_size:len(X)]

    modelMLP = MLPRegressor(hidden_layer_sizes= (50,))
    modelOLS = LinearRegression()
    model_MLP = modelMLP.fit(X_train, Y_train)
    model_OLS = modelOLS.fit(X_train, Y_train)

    Y_predMLP = pd.DataFrame(model_MLP.predict(X_validation), index=Y_validation.index,
                          columns=Y_validation.columns)

    Y_predOLS = pd.DataFrame(model_OLS.predict(X_validation), index=Y_validation.index,
                          columns=Y_validation.columns)


.. code:: ipython3

    pd.DataFrame({'Actual : 1m': Y_validation.loc[:, 'DGS1MO_pred'],
                  'Prediction MLP 1m': Y_predMLP.loc[:, 'DGS1MO_pred'],
                  'Prediction OLS 1m': Y_predOLS.loc[:, 'DGS1MO_pred']}).plot(figsize=(10,5))

    pd.DataFrame({'Actual : 5yr': Y_validation.loc[:, 'DGS5_pred'],
                  'Prediction MLP 5yr': Y_predMLP.loc[:, 'DGS5_pred'],
                  'Prediction OLS 5yr': Y_predOLS.loc[:, 'DGS5_pred']}).plot(figsize=(10,5))

    pd.DataFrame({'Actual : 30yr': Y_validation.loc[:, 'DGS30_pred'],
                  'Prediction MLP 30yr': Y_predMLP.loc[:, 'DGS30_pred'],
                  'Prediction OLS 30yr': Y_predOLS.loc[:, 'DGS30_pred']}).plot(figsize=(10,5))






.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x25e44811860>




.. image:: output_58_1.png



.. image:: output_58_2.png



.. image:: output_58_3.png


Overall, the regression and MLP are comparable, however, for 1m tenor,
the fitting with MLP is slighly poor as compared to the regression.
However,the multitask learning with neural network is more intuitive for
modeling many time series simultaneousl

Summary
~~~~~~~

The linear regression model, despite its simplicity, is a tough
benchmark to beat for such one step ahead forecasting, given the
dominant characteristic of the last available value of the variable to
predict. The ANN results in this case study are comparable to the linear
regression models.

The good thing about ANN is that it is more flexible to changing market
conditions. Also, ANN models can be enhanced by performing grid search
on several other hyperparameters and using recurrent neural network such
as LSTM.
