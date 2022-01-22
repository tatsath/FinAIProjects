.. _supervised_reg:

Supervised - Classification
===============

Template for Supervised Classification

Content
------------------------------------------------


1. Problem Statement
------------------------------------------------

Our goal in this jupyter notebook is to under the following. A sample
problem of stock price prediction in presented in this case study. - How
to work through a predictive modeling problem end-to-end. This notebook
is applicable both for regression and classification problems. - How to
use data transforms to improve model performance. - How to use algorithm
tuning to improve model performance. - How to use ensemble methods and
tuning of ensemble methods to improve model performance. - How to use
deep Learning methods. - Following Models are implemented

::

   * Linear Regression
   * Lasso
   * Elastic Net
   * KNN
   * Decision Tree (CART)
   * Support Vector Machine
   * Ada Boost
   * Gradient Boosting Method
   * Random Forest
   * Extra Trees
   * Neural Network - Shallow - Using sklearn
   * Deep Neural Network - Using Keras

-  Time Series Models

   -  ARIMA Model
   -  LSTM - Using Keras

2. Getting Started- Loading the data and python packages
------------------------------------------------

 ## 2.1. Loading the python packages

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


.. parsed-literal::

    Using TensorFlow backend.


 ## 2.2. Loading the Data

.. code:: ipython3

    # Get the data by webscapping using pandas datareader
    return_period = 21


    stk_tickers = ['MSFT', 'IBM', 'GOOGL']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred')
    idx_data = web.DataReader(idx_tickers, 'fred')

    Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1]+'_pred'

    X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X4 = pd.concat([Y.diff(i) for i in [21, 63, 126,252]], axis=1).dropna()
    X4.columns = ['1M', '3M', '6M', '1Y']

    X = pd.concat([X1, X2, X3, X4], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna()
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

.. code:: ipython3

    type(dataset)




.. parsed-literal::

    pandas.core.frame.DataFrame



Converting the data to supervised regression format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the predictor variables are changed to lagged variable, as the t-1
value of the lagged variable will be used for prediction.

.. code:: ipython3

    def series_to_supervised(data, lag=1):
        n_vars = data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(lag, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg = pd.DataFrame(data.iloc[:,0]).join(agg)
        agg.dropna(inplace=True)
        return agg

.. code:: ipython3

    dataset= series_to_supervised(dataset,1)

 # 3. Exploratory Data Analysis

 ## 3.1. Descriptive Statistics

.. code:: ipython3

    # shape
    dataset.shape




.. parsed-literal::

    (2252, 13)



.. code:: ipython3

    # peek at data
    pd.set_option('display.width', 100)
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
          <th>MSFT_pred</th>
          <th>MSFT_pred(t-1)</th>
          <th>GOOGL(t-1)</th>
          <th>IBM(t-1)</th>
          <th>DEXJPUS(t-1)</th>
          <th>DEXUSUK(t-1)</th>
          <th>SP500(t-1)</th>
          <th>DJIA(t-1)</th>
          <th>VIXCLS(t-1)</th>
          <th>1M(t-1)</th>
          <th>3M(t-1)</th>
          <th>6M(t-1)</th>
          <th>1Y(t-1)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-04</th>
          <td>-0.015788</td>
          <td>-0.001431</td>
          <td>0.055329</td>
          <td>0.015718</td>
          <td>-0.014243</td>
          <td>-0.014930</td>
          <td>0.037784</td>
          <td>0.025045</td>
          <td>-0.022460</td>
          <td>-0.041166</td>
          <td>-0.137312</td>
          <td>-0.078563</td>
          <td>0.076487</td>
        </tr>
        <tr>
          <th>2011-01-05</th>
          <td>-0.008248</td>
          <td>-0.015788</td>
          <td>0.049571</td>
          <td>0.015426</td>
          <td>-0.008988</td>
          <td>-0.006013</td>
          <td>0.037769</td>
          <td>0.028544</td>
          <td>-0.036162</td>
          <td>-0.054624</td>
          <td>-0.120203</td>
          <td>-0.058879</td>
          <td>0.090434</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # types
    pd.set_option('display.max_rows', 500)
    dataset.dtypes




.. parsed-literal::

    MSFT_pred         float64
    MSFT_pred(t-1)    float64
    GOOGL(t-1)        float64
    IBM(t-1)          float64
    DEXJPUS(t-1)      float64
    DEXUSUK(t-1)      float64
    SP500(t-1)        float64
    DJIA(t-1)         float64
    VIXCLS(t-1)       float64
    1M(t-1)           float64
    3M(t-1)           float64
    6M(t-1)           float64
    1Y(t-1)           float64
    dtype: object



.. code:: ipython3

    # describe data
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
          <th>MSFT_pred</th>
          <th>MSFT_pred(t-1)</th>
          <th>GOOGL(t-1)</th>
          <th>IBM(t-1)</th>
          <th>DEXJPUS(t-1)</th>
          <th>DEXUSUK(t-1)</th>
          <th>SP500(t-1)</th>
          <th>DJIA(t-1)</th>
          <th>VIXCLS(t-1)</th>
          <th>1M(t-1)</th>
          <th>3M(t-1)</th>
          <th>6M(t-1)</th>
          <th>1Y(t-1)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>2252.000</td>
          <td>2252.000</td>
          <td>2252.000</td>
          <td>2252.000</td>
          <td>2.252e+03</td>
          <td>2.252e+03</td>
          <td>2252.000</td>
          <td>2252.000</td>
          <td>2252.000</td>
          <td>2.252e+03</td>
          <td>2.252e+03</td>
          <td>2252.000</td>
          <td>2252.000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>0.019</td>
          <td>0.019</td>
          <td>0.014</td>
          <td>0.001</td>
          <td>2.056e-03</td>
          <td>-1.915e-03</td>
          <td>0.008</td>
          <td>0.007</td>
          <td>0.005</td>
          <td>4.086e-04</td>
          <td>5.506e-04</td>
          <td>0.001</td>
          <td>0.004</td>
        </tr>
        <tr>
          <th>std</th>
          <td>0.058</td>
          <td>0.058</td>
          <td>0.067</td>
          <td>0.063</td>
          <td>2.503e-02</td>
          <td>2.374e-02</td>
          <td>0.043</td>
          <td>0.045</td>
          <td>0.265</td>
          <td>8.737e-02</td>
          <td>8.570e-02</td>
          <td>0.077</td>
          <td>0.081</td>
        </tr>
        <tr>
          <th>min</th>
          <td>-0.302</td>
          <td>-0.302</td>
          <td>-0.351</td>
          <td>-0.461</td>
          <td>-8.290e-02</td>
          <td>-1.227e-01</td>
          <td>-0.400</td>
          <td>-0.444</td>
          <td>-0.827</td>
          <td>-4.272e-01</td>
          <td>-3.467e-01</td>
          <td>-0.327</td>
          <td>-0.410</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>-0.014</td>
          <td>-0.014</td>
          <td>-0.028</td>
          <td>-0.031</td>
          <td>-1.294e-02</td>
          <td>-1.556e-02</td>
          <td>-0.009</td>
          <td>-0.010</td>
          <td>-0.148</td>
          <td>-5.193e-02</td>
          <td>-4.780e-02</td>
          <td>-0.047</td>
          <td>-0.048</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>0.023</td>
          <td>0.023</td>
          <td>0.017</td>
          <td>0.004</td>
          <td>9.865e-04</td>
          <td>-7.240e-04</td>
          <td>0.014</td>
          <td>0.012</td>
          <td>-0.021</td>
          <td>5.407e-05</td>
          <td>1.940e-03</td>
          <td>-0.002</td>
          <td>0.004</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>0.052</td>
          <td>0.052</td>
          <td>0.053</td>
          <td>0.039</td>
          <td>1.489e-02</td>
          <td>1.384e-02</td>
          <td>0.030</td>
          <td>0.031</td>
          <td>0.120</td>
          <td>4.945e-02</td>
          <td>5.187e-02</td>
          <td>0.045</td>
          <td>0.056</td>
        </tr>
        <tr>
          <th>max</th>
          <td>0.244</td>
          <td>0.244</td>
          <td>0.271</td>
          <td>0.230</td>
          <td>1.023e-01</td>
          <td>6.963e-02</td>
          <td>0.203</td>
          <td>0.214</td>
          <td>1.799</td>
          <td>5.407e-01</td>
          <td>3.619e-01</td>
          <td>0.275</td>
          <td>0.230</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 3.2. Data Visualization

.. code:: ipython3

    # histograms
    dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
    pyplot.show()



.. image:: output_22_0.png


.. code:: ipython3

    # density
    dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True, fontsize=1, figsize=(15,15))
    pyplot.show()



.. image:: output_23_0.png


.. code:: ipython3

    #Box and Whisker Plots
    dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(15,15))
    pyplot.show()



.. image:: output_24_0.png


.. code:: ipython3

    # correlation
    correlation = dataset.corr()
    pyplot.figure(figsize=(15,15))
    pyplot.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x21b92b917b8>




.. image:: output_25_1.png


.. code:: ipython3

    # Scatterplot Matrix
    from pandas.plotting import scatter_matrix
    pyplot.figure(figsize=(15,15))
    scatter_matrix(dataset,figsize=(12,12))
    pyplot.show()



.. parsed-literal::

    <Figure size 1080x1080 with 0 Axes>



.. image:: output_26_1.png


 ## 3.3. Time Series Analysis

Time series broken down into different time series comonent

.. code:: ipython3

    Y= dataset["MSFT_pred"]
    res = sm.tsa.seasonal_decompose(Y,freq=365)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()



.. image:: output_29_0.png


 ## 4. Data Preparation

 ## 4.1. Data Cleaning Check for the NAs in the rows, either drop them
or fill them with the mean of the column

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())


.. parsed-literal::

    Null Values = False


Given that there are null values drop the rown contianing the null
values.

.. code:: ipython3

    # Drop the rows containing NA
    #dataset.dropna(axis=0)
    # Fill na with 0
    #dataset.fillna('0')

    #Filling the NAs with the mean of the column.
    #dataset['col'] = dataset['col'].fillna(dataset['col'].mean())

 ## 4.3. Feature Selection Statistical tests can be used to select those
features that have the strongest relationship with the output
variable.The scikit-learn library provides the SelectKBest class that
can be used with a suite of different statistical tests to select a
specific number of features. The example below uses the chi-squared
(chi²) statistical test for non-negative features to select 10 of the
best features from the Dataset.

.. code:: ipython3

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    bestfeatures = SelectKBest(k=5)
    bestfeatures




.. parsed-literal::

    SelectKBest(k=5, score_func=<function f_classif at 0x0000021B972962F0>)



.. code:: ipython3

    type(dataset)




.. parsed-literal::

    pandas.core.frame.DataFrame



.. code:: ipython3

    Y= dataset["MSFT_pred"]
    X = dataset.loc[:, dataset.columns != 'MSFT_pred']
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features


.. parsed-literal::

                 Specs    Score
    0   MSFT_pred(t-1)  667.074
    1       GOOGL(t-1)   15.767
    10         6M(t-1)    7.466
    9          3M(t-1)    6.491
    4     DEXUSUK(t-1)    3.361
    5       SP500(t-1)    1.716
    8          1M(t-1)    1.656
    7      VIXCLS(t-1)    1.441
    11         1Y(t-1)    1.197
    6        DJIA(t-1)    1.175


As it can be seen from the result above that t-1 is an important feature



 ## 4.4. Data Transformation

 ### 4.4.1. Rescale Data When your data is comprised of attributes with
varying scales, many machine learning algorithms can benefit from
rescaling the attributes to all have the same scale. Often this is
referred to as normalization and attributes are often rescaled into the
range between 0 and 1.

.. code:: ipython3

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = pd.DataFrame(scaler.fit_transform(X))
    # summarize transformed data
    rescaledX.head(5)




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
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10</th>
          <th>11</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.551</td>
          <td>0.653</td>
          <td>0.690</td>
          <td>0.371</td>
          <td>0.560</td>
          <td>0.726</td>
          <td>0.713</td>
          <td>0.306</td>
          <td>0.399</td>
          <td>0.296</td>
          <td>0.413</td>
          <td>0.760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.525</td>
          <td>0.644</td>
          <td>0.689</td>
          <td>0.399</td>
          <td>0.607</td>
          <td>0.726</td>
          <td>0.719</td>
          <td>0.301</td>
          <td>0.385</td>
          <td>0.320</td>
          <td>0.446</td>
          <td>0.781</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.538</td>
          <td>0.648</td>
          <td>0.687</td>
          <td>0.459</td>
          <td>0.537</td>
          <td>0.734</td>
          <td>0.723</td>
          <td>0.294</td>
          <td>0.389</td>
          <td>0.329</td>
          <td>0.454</td>
          <td>0.773</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.514</td>
          <td>0.635</td>
          <td>0.713</td>
          <td>0.382</td>
          <td>0.538</td>
          <td>0.724</td>
          <td>0.718</td>
          <td>0.307</td>
          <td>0.347</td>
          <td>0.331</td>
          <td>0.418</td>
          <td>0.753</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.533</td>
          <td>0.633</td>
          <td>0.696</td>
          <td>0.395</td>
          <td>0.580</td>
          <td>0.715</td>
          <td>0.716</td>
          <td>0.312</td>
          <td>0.379</td>
          <td>0.350</td>
          <td>0.509</td>
          <td>0.764</td>
        </tr>
      </tbody>
    </table>
    </div>



 ### 4.4.2. Standardize Data Standardization is a useful technique to
transform attributes with a Gaussian distribution and differing means
and standard deviations to a standard Gaussian distribution with a mean
of 0 and a standard deviation of 1.

.. code:: ipython3

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    StandardisedX = pd.DataFrame(scaler.fit_transform(X))
    # summarize transformed data
    StandardisedX.head(5)




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
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10</th>
          <th>11</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.357</td>
          <td>0.624</td>
          <td>0.229</td>
          <td>-0.651</td>
          <td>-0.548</td>
          <td>0.695</td>
          <td>0.408</td>
          <td>-0.104</td>
          <td>-0.476</td>
          <td>-1.609</td>
          <td>-1.041</td>
          <td>0.889</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.604</td>
          <td>0.538</td>
          <td>0.224</td>
          <td>-0.441</td>
          <td>-0.173</td>
          <td>0.694</td>
          <td>0.486</td>
          <td>-0.155</td>
          <td>-0.630</td>
          <td>-1.409</td>
          <td>-0.784</td>
          <td>1.061</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.474</td>
          <td>0.570</td>
          <td>0.203</td>
          <td>0.004</td>
          <td>-0.740</td>
          <td>0.797</td>
          <td>0.552</td>
          <td>-0.228</td>
          <td>-0.583</td>
          <td>-1.330</td>
          <td>-0.719</td>
          <td>0.999</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-0.707</td>
          <td>0.454</td>
          <td>0.482</td>
          <td>-0.565</td>
          <td>-0.733</td>
          <td>0.663</td>
          <td>0.477</td>
          <td>-0.092</td>
          <td>-1.056</td>
          <td>-1.315</td>
          <td>-1.001</td>
          <td>0.836</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.526</td>
          <td>0.439</td>
          <td>0.299</td>
          <td>-0.470</td>
          <td>-0.385</td>
          <td>0.533</td>
          <td>0.439</td>
          <td>-0.043</td>
          <td>-0.695</td>
          <td>-1.156</td>
          <td>-0.287</td>
          <td>0.922</td>
        </tr>
      </tbody>
    </table>
    </div>



 ### 4.4.1. Normalize Data Normalizing in scikit-learn refers to
rescaling each observation (row) to have a length of 1 (called a unit
norm or a vector with the length of 1 in linear algebra).

.. code:: ipython3

    from sklearn.preprocessing import Normalizer
    scaler = Normalizer().fit(X)
    NormalizedX = pd.DataFrame(scaler.fit_transform(X))
    # summarize transformed data
    NormalizedX.head(5)




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
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10</th>
          <th>11</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.007</td>
          <td>0.281</td>
          <td>0.080</td>
          <td>-0.072</td>
          <td>-0.076</td>
          <td>0.192</td>
          <td>0.127</td>
          <td>-0.114</td>
          <td>-0.209</td>
          <td>-0.696</td>
          <td>-0.398</td>
          <td>0.388</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.084</td>
          <td>0.262</td>
          <td>0.082</td>
          <td>-0.048</td>
          <td>-0.032</td>
          <td>0.200</td>
          <td>0.151</td>
          <td>-0.191</td>
          <td>-0.289</td>
          <td>-0.636</td>
          <td>-0.312</td>
          <td>0.479</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.044</td>
          <td>0.277</td>
          <td>0.076</td>
          <td>0.012</td>
          <td>-0.104</td>
          <td>0.226</td>
          <td>0.169</td>
          <td>-0.297</td>
          <td>-0.271</td>
          <td>-0.608</td>
          <td>-0.289</td>
          <td>0.458</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-0.111</td>
          <td>0.224</td>
          <td>0.162</td>
          <td>-0.062</td>
          <td>-0.099</td>
          <td>0.186</td>
          <td>0.144</td>
          <td>-0.099</td>
          <td>-0.469</td>
          <td>-0.573</td>
          <td>-0.386</td>
          <td>0.369</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.072</td>
          <td>0.275</td>
          <td>0.129</td>
          <td>-0.062</td>
          <td>-0.071</td>
          <td>0.197</td>
          <td>0.170</td>
          <td>-0.041</td>
          <td>-0.387</td>
          <td>-0.632</td>
          <td>-0.134</td>
          <td>0.508</td>
        </tr>
      </tbody>
    </table>
    </div>



 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split

.. code:: ipython3

    # split out validation dataset for the end

    validation_size = 0.2

    #In case the data is not dependent on the time series, then train and test split randomly
    seed = 7
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    #In case the data is not dependent on the time series, then train and test split should be done based on sequential sample
    #This can be done by selecting an arbitrary split point in the ordered list of observations and creating two new datasets.

    train_size = int(len(X) * (1-validation_size))
    X_train, X_validation = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_validation = Y[0:train_size], Y[train_size:len(X)]

 ## 5.2. Test Options and Evaluation Metrics

.. code:: ipython3

    # test options for regression
    num_folds = 10
    scoring = 'neg_mean_squared_error'
    #scoring ='neg_mean_absolute_error'
    #scoring = 'r2'

 ## 5.3. Compare Models and Algorithms

 ### 5.3.1. Common Models

.. code:: ipython3

    # spot check the algorithms
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))
    #Neural Network
    models.append(('MLP', MLPRegressor()))

 ### 5.3.2. Ensemble Models

.. code:: ipython3

    #Ensable Models
    # Boosting methods
    models.append(('ABR', AdaBoostRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))
    # Bagging methods
    models.append(('RFR', RandomForestRegressor()))
    models.append(('ETR', ExtraTreesRegressor()))

 ### 5.3.3. Deep Learning Model-NN Regressor

.. code:: ipython3

    #Running deep learning models and performing cross validation takes time
    #Set the following Flag to 0 if the Deep LEarning Models Flag has to be disabled
    EnableDeepLearningRegreesorFlag = 0

    def create_model(neurons=12, activation='relu', learn_rate = 0.01, momentum=0):
            # create model
            model = Sequential()
            model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
            #The number of hidden layers can be increased
            model.add(Dense(2, activation=activation))
            # Final output layer
            model.add(Dense(1, kernel_initializer='normal'))
            # Compile model
            optimizer = SGD(lr=learn_rate, momentum=momentum)
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

.. code:: ipython3

    #Add Deep Learning Regressor
    if ( EnableDeepLearningRegreesorFlag == 1):
        models.append(('DNN', KerasRegressor(build_fn=create_model, epochs=100, batch_size=100, verbose=1)))


K-folds cross validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        #converted mean square error to positive. The lower the beter
        cv_results = -1* cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


.. parsed-literal::

    LR: 0.000419 (0.000159)
    LASSO: 0.003024 (0.001482)
    EN: 0.003024 (0.001482)
    KNN: 0.000934 (0.000363)
    CART: 0.000988 (0.000280)
    SVR: 0.001448 (0.000906)
    MLP: 0.000734 (0.000242)
    ABR: 0.000577 (0.000199)
    GBR: 0.000460 (0.000179)
    RFR: 0.000473 (0.000185)
    ETR: 0.000472 (0.000192)


Algorithm comparison
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15,8)
    pyplot.show()



.. image:: output_64_0.png


The chart shows MSE. Lower the MSE, better is the model performance.

 ## 5.4. Time Series based Models- ARIMA and LSTM

 ### 5.4.1 Time Series Model - ARIMA Model

.. code:: ipython3

    #Preparing data for the ARIMAX Model, seperating endogeneous and exogenous variables
    X_train_ARIMA=X_train.drop(['MSFT_pred(t-1)'], axis = 'columns' ).dropna()
    X_validation_ARIMA=X_validation.drop(['MSFT_pred(t-1)'], axis = 'columns' ).dropna()
    tr_len = len(X_train_ARIMA)
    te_len = len(X_validation_ARIMA)
    to_len = len (X)

.. code:: ipython3

    from statsmodels.tsa.arima_model import ARIMA
    #from statsmodels.tsa.statespace.sarimax import SARIMAX

    from sklearn.metrics import mean_squared_error

    modelARIMA=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=[1,0,0])
    #modelARIMA= SARIMAX(Y_train,order=(1,1,0),seasonal_order=[1,0,0,0],exog = X_train_ARIMA)

    model_fit = modelARIMA.fit()
    #print(model_fit.summary())

.. code:: ipython3

    error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
    predicted = model_fit.predict(start = tr_len -1 ,end = to_len -1, exog = X_validation_ARIMA)[1:]
    error_Test_ARIMA = mean_squared_error(Y_validation,predicted)
    error_Test_ARIMA




.. parsed-literal::

    0.0051007878797309026



.. code:: ipython3

    #Add Cross validation if possible
    # #model = build_model(_alpha=1.0, _l1_ratio=0.3)
    # from sklearn.model_selection import TimeSeriesSplit
    # tscv = TimeSeriesSplit(n_splits=5)
    # scores = cross_val_score(modelARIMA, X_train, Y_train, cv=tscv, scoring=scoring)

 ### 5.4.2 LSTM Model

The data needs to be in 3D format for the LSTM model. So, Performing the
data transform.

.. code:: ipython3

    X_train_LSTM, X_validation_LSTM = np.array(X_train), np.array(X_validation)
    Y_train_LSTM, Y_validation_LSTM = np.array(Y_train), np.array(Y_validation)
    X_train_LSTM= X_train_LSTM.reshape((X_train_LSTM.shape[0], 1, X_train_LSTM.shape[1]))
    X_validation_LSTM= X_validation_LSTM.reshape((X_validation_LSTM.shape[0], 1, X_validation_LSTM.shape[1]))
    print(X_train_LSTM.shape, Y_train_LSTM.shape, X_validation_LSTM.shape, Y_validation_LSTM.shape)


.. parsed-literal::

    (1801, 1, 12) (1801,) (451, 1, 12) (451,)


.. code:: ipython3

    # design network
    from matplotlib import pyplot

    def create_LSTMmodel(neurons=12, learn_rate = 0.01, momentum=0):
            # create model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
        #More number of cells can be added if needed
        model.add(Dense(1))
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(loss='mse', optimizer='adam')
        return model
    LSTMModel = create_LSTMmodel(12, learn_rate = 0.01, momentum=0)
    LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_validation_LSTM, Y_validation_LSTM),epochs=50, batch_size=72, verbose=0, shuffle=False)# plot history



.. parsed-literal::

    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.


.. code:: ipython3

    #Visual plot to check if the error is reducing
    pyplot.plot(LSTMModel_fit.history['loss'], label='train')
    pyplot.plot(LSTMModel_fit.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()



.. image:: output_76_0.png


.. code:: ipython3

    error_Training_LSTM = mean_squared_error(Y_train_LSTM, LSTMModel.predict(X_train_LSTM))
    predicted = LSTMModel.predict(X_validation_LSTM)
    error_Test_LSTM = mean_squared_error(Y_validation,predicted)
    error_Test_LSTM




.. parsed-literal::

    0.000906767112032725



Overall Comparison of all the algorithms ( including Time Series Algorithms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # compare algorithms
    results.append(error_Test_ARIMA)
    results.append(error_Test_LSTM)
    names.append("ARIMA")
    names.append("LSTM")
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison-Post Time Series')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15,8)
    pyplot.show()



.. image:: output_79_0.png


Grid Search uses Cross validation which isn’t appropriate for the time
series models such as LSTM

 # 6. Model Tuning and Grid Search This section shown the Grid search
for all the Machine Learning and time series models mentioned in the
book.

 ### 6.1. Common Regression, Ensemble and DeepNNRegressor Grid Search

.. code:: ipython3

    # 1. Grid search : LinearRegression
    '''
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    '''
    param_grid = {'fit_intercept': [True, False]}
    model = LinearRegression()
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

    Best: -0.000419 using {'fit_intercept': True}
    -0.000419 (0.000159) with: {'fit_intercept': True}
    -0.000419 (0.000158) with: {'fit_intercept': False}


.. code:: ipython3

    # 2. Grid search : Lasso
    '''
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.
    '''
    param_grid = {'alpha': [0.01, 0.1, 0.3, 0.7, 1, 1.5, 3, 5]}
    model = Lasso()
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

    Best: -0.003024 using {'alpha': 0.01}
    -0.003024 (0.001482) with: {'alpha': 0.01}
    -0.003024 (0.001482) with: {'alpha': 0.1}
    -0.003024 (0.001482) with: {'alpha': 0.3}
    -0.003024 (0.001482) with: {'alpha': 0.7}
    -0.003024 (0.001482) with: {'alpha': 1}
    -0.003024 (0.001482) with: {'alpha': 1.5}
    -0.003024 (0.001482) with: {'alpha': 3}
    -0.003024 (0.001482) with: {'alpha': 5}


.. code:: ipython3

    # 3. Grid Search : ElasticNet
    '''
    alpha : float, optional
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.
    '''
    param_grid = {'alpha': [0.01, 0.1, 0.3, 0.7, 1, 1.5, 3, 5],
                  'l1_ratio': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]}
    model = ElasticNet()
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

    Best: -0.001091 using {'alpha': 0.01, 'l1_ratio': 0.01}
    -0.001091 (0.000493) with: {'alpha': 0.01, 'l1_ratio': 0.01}
    -0.001526 (0.000750) with: {'alpha': 0.01, 'l1_ratio': 0.1}
    -0.002986 (0.001506) with: {'alpha': 0.01, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 0.01, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 0.01, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 0.01, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 0.01, 'l1_ratio': 0.99}
    -0.002616 (0.001297) with: {'alpha': 0.1, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 0.1, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 0.1, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 0.1, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 0.1, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 0.1, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 0.1, 'l1_ratio': 0.99}
    -0.003022 (0.001483) with: {'alpha': 0.3, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 0.3, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 0.3, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 0.3, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 0.3, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 0.3, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 0.3, 'l1_ratio': 0.99}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 0.7, 'l1_ratio': 0.99}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 1, 'l1_ratio': 0.99}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 1.5, 'l1_ratio': 0.99}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 3, 'l1_ratio': 0.99}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.01}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.1}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.3}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.5}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.7}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.9}
    -0.003024 (0.001482) with: {'alpha': 5, 'l1_ratio': 0.99}


.. code:: ipython3


    # 4. Grid search : KNeighborsRegressor
    '''
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    '''
    param_grid = {'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21]}
    model = KNeighborsRegressor()
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

    Best: -0.000860 using {'n_neighbors': 17}
    -0.001571 (0.000373) with: {'n_neighbors': 1}
    -0.001040 (0.000358) with: {'n_neighbors': 3}
    -0.000934 (0.000363) with: {'n_neighbors': 5}
    -0.000886 (0.000349) with: {'n_neighbors': 7}
    -0.000877 (0.000358) with: {'n_neighbors': 9}
    -0.000871 (0.000353) with: {'n_neighbors': 11}
    -0.000865 (0.000361) with: {'n_neighbors': 13}
    -0.000864 (0.000358) with: {'n_neighbors': 15}
    -0.000860 (0.000361) with: {'n_neighbors': 17}
    -0.000865 (0.000365) with: {'n_neighbors': 19}
    -0.000864 (0.000372) with: {'n_neighbors': 21}


.. code:: ipython3

    # 5. Grid search : DecisionTreeRegressor
    '''
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    '''
    param_grid={'min_samples_split': [2,3,4,5,6,7,8,9,10]}
    model = DecisionTreeRegressor()
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

    Best: -0.000780 using {'min_samples_split': 10}
    -0.000928 (0.000256) with: {'min_samples_split': 2}
    -0.000932 (0.000322) with: {'min_samples_split': 3}
    -0.000919 (0.000266) with: {'min_samples_split': 4}
    -0.000907 (0.000300) with: {'min_samples_split': 5}
    -0.000878 (0.000240) with: {'min_samples_split': 6}
    -0.000866 (0.000266) with: {'min_samples_split': 7}
    -0.000872 (0.000249) with: {'min_samples_split': 8}
    -0.000826 (0.000210) with: {'min_samples_split': 9}
    -0.000780 (0.000196) with: {'min_samples_split': 10}


.. code:: ipython3

    # 6. Grid search : SVR
    '''
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.
    '''
    param_grid={'C': [0.01, 0.03,0.1,0.3,1,3,10,30,100],
                'gamma': [0.001, 0.01, 0.1, 1]},
                #'epslion': [0.01, 0.1, 1]}
    model = SVR()
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

    Best: -0.000968 using {'C': 100, 'gamma': 0.01}
    -0.002999 (0.001492) with: {'C': 0.01, 'gamma': 0.001}
    -0.002928 (0.001476) with: {'C': 0.01, 'gamma': 0.01}
    -0.002514 (0.001316) with: {'C': 0.01, 'gamma': 0.1}
    -0.001806 (0.000995) with: {'C': 0.01, 'gamma': 1}
    -0.002982 (0.001488) with: {'C': 0.03, 'gamma': 0.001}
    -0.002792 (0.001428) with: {'C': 0.03, 'gamma': 0.01}
    -0.002106 (0.001121) with: {'C': 0.03, 'gamma': 0.1}
    -0.001586 (0.000873) with: {'C': 0.03, 'gamma': 1}
    -0.002928 (0.001476) with: {'C': 0.1, 'gamma': 0.001}
    -0.002509 (0.001311) with: {'C': 0.1, 'gamma': 0.01}
    -0.001779 (0.000943) with: {'C': 0.1, 'gamma': 0.1}
    -0.001240 (0.000610) with: {'C': 0.1, 'gamma': 1}
    -0.002791 (0.001427) with: {'C': 0.3, 'gamma': 0.001}
    -0.002097 (0.001120) with: {'C': 0.3, 'gamma': 0.01}
    -0.001542 (0.000810) with: {'C': 0.3, 'gamma': 0.1}
    -0.001218 (0.000536) with: {'C': 0.3, 'gamma': 1}
    -0.002508 (0.001310) with: {'C': 1, 'gamma': 0.001}
    -0.001776 (0.000941) with: {'C': 1, 'gamma': 0.01}
    -0.001208 (0.000574) with: {'C': 1, 'gamma': 0.1}
    -0.001177 (0.000493) with: {'C': 1, 'gamma': 1}
    -0.002097 (0.001114) with: {'C': 3, 'gamma': 0.001}
    -0.001546 (0.000818) with: {'C': 3, 'gamma': 0.01}
    -0.001132 (0.000492) with: {'C': 3, 'gamma': 0.1}
    -0.001177 (0.000493) with: {'C': 3, 'gamma': 1}
    -0.001776 (0.000941) with: {'C': 10, 'gamma': 0.001}
    -0.001179 (0.000590) with: {'C': 10, 'gamma': 0.01}
    -0.001065 (0.000409) with: {'C': 10, 'gamma': 0.1}
    -0.001177 (0.000493) with: {'C': 10, 'gamma': 1}
    -0.001549 (0.000823) with: {'C': 30, 'gamma': 0.001}
    -0.001151 (0.000540) with: {'C': 30, 'gamma': 0.01}
    -0.001065 (0.000409) with: {'C': 30, 'gamma': 0.1}
    -0.001177 (0.000493) with: {'C': 30, 'gamma': 1}
    -0.001178 (0.000594) with: {'C': 100, 'gamma': 0.001}
    -0.000968 (0.000413) with: {'C': 100, 'gamma': 0.01}
    -0.001065 (0.000409) with: {'C': 100, 'gamma': 0.1}
    -0.001177 (0.000493) with: {'C': 100, 'gamma': 1}


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

    Best: -0.000801 using {'hidden_layer_sizes': (50,)}
    -0.001169 (0.000496) with: {'hidden_layer_sizes': (20,)}
    -0.000801 (0.000337) with: {'hidden_layer_sizes': (50,)}
    -0.000994 (0.000372) with: {'hidden_layer_sizes': (20, 20)}
    -0.000880 (0.000292) with: {'hidden_layer_sizes': (20, 30, 20)}


.. code:: ipython3

    # 8. Grid search : RandomForestRegressor
    '''
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    '''
    param_grid = {'n_estimators': [50,100,150,200,250,300,350,400]}
    model = RandomForestRegressor()
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

    Best: -0.000470 using {'n_estimators': 400}
    -0.000479 (0.000189) with: {'n_estimators': 50}
    -0.000470 (0.000182) with: {'n_estimators': 100}
    -0.000471 (0.000183) with: {'n_estimators': 150}
    -0.000470 (0.000182) with: {'n_estimators': 200}
    -0.000471 (0.000183) with: {'n_estimators': 250}
    -0.000473 (0.000185) with: {'n_estimators': 300}
    -0.000471 (0.000180) with: {'n_estimators': 350}
    -0.000470 (0.000181) with: {'n_estimators': 400}


.. code:: ipython3


    # 9. Grid search : GradientBoostingRegressor
    '''
    n_estimators:

        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    '''
    param_grid = {'n_estimators': [50,100,150,200,250,300,350,400]}
    model = GradientBoostingRegressor(random_state=seed)
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

    Best: -0.000446 using {'n_estimators': 50}
    -0.000446 (0.000174) with: {'n_estimators': 50}
    -0.000461 (0.000182) with: {'n_estimators': 100}
    -0.000474 (0.000186) with: {'n_estimators': 150}
    -0.000484 (0.000191) with: {'n_estimators': 200}
    -0.000492 (0.000193) with: {'n_estimators': 250}
    -0.000498 (0.000193) with: {'n_estimators': 300}
    -0.000505 (0.000196) with: {'n_estimators': 350}
    -0.000511 (0.000195) with: {'n_estimators': 400}


.. code:: ipython3

    # 10. Grid search : ExtraTreesRegressor
    '''
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    '''
    param_grid = {'n_estimators': [50,100,150,200,250,300,350,400]}
    model = ExtraTreesRegressor(random_state=seed)
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

    Best: -0.000470 using {'n_estimators': 400}
    -0.000472 (0.000186) with: {'n_estimators': 50}
    -0.000473 (0.000186) with: {'n_estimators': 100}
    -0.000474 (0.000189) with: {'n_estimators': 150}
    -0.000472 (0.000189) with: {'n_estimators': 200}
    -0.000471 (0.000190) with: {'n_estimators': 250}
    -0.000471 (0.000190) with: {'n_estimators': 300}
    -0.000470 (0.000189) with: {'n_estimators': 350}
    -0.000470 (0.000188) with: {'n_estimators': 400}


.. code:: ipython3

    # 11. Grid search : AdaBoostRegre
    '''
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    '''
    param_grid = {'n_estimators': [50,100,150,200,250,300,350,400],
                 'learning_rate': [1, 2, 3]}
    model = AdaBoostRegressor(random_state=seed)
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

    Best: -0.000574 using {'learning_rate': 1, 'n_estimators': 50}
    -0.000574 (0.000189) with: {'learning_rate': 1, 'n_estimators': 50}
    -0.000607 (0.000195) with: {'learning_rate': 1, 'n_estimators': 100}
    -0.000613 (0.000181) with: {'learning_rate': 1, 'n_estimators': 150}
    -0.000625 (0.000180) with: {'learning_rate': 1, 'n_estimators': 200}
    -0.000634 (0.000180) with: {'learning_rate': 1, 'n_estimators': 250}
    -0.000640 (0.000182) with: {'learning_rate': 1, 'n_estimators': 300}
    -0.000641 (0.000184) with: {'learning_rate': 1, 'n_estimators': 350}
    -0.000639 (0.000182) with: {'learning_rate': 1, 'n_estimators': 400}
    -0.000606 (0.000191) with: {'learning_rate': 2, 'n_estimators': 50}
    -0.000609 (0.000189) with: {'learning_rate': 2, 'n_estimators': 100}
    -0.000610 (0.000188) with: {'learning_rate': 2, 'n_estimators': 150}
    -0.000620 (0.000189) with: {'learning_rate': 2, 'n_estimators': 200}
    -0.000620 (0.000189) with: {'learning_rate': 2, 'n_estimators': 250}
    -0.000621 (0.000184) with: {'learning_rate': 2, 'n_estimators': 300}
    -0.000625 (0.000182) with: {'learning_rate': 2, 'n_estimators': 350}
    -0.000623 (0.000185) with: {'learning_rate': 2, 'n_estimators': 400}
    -0.000630 (0.000185) with: {'learning_rate': 3, 'n_estimators': 50}
    -0.000613 (0.000184) with: {'learning_rate': 3, 'n_estimators': 100}
    -0.000616 (0.000184) with: {'learning_rate': 3, 'n_estimators': 150}
    -0.000613 (0.000192) with: {'learning_rate': 3, 'n_estimators': 200}
    -0.000617 (0.000187) with: {'learning_rate': 3, 'n_estimators': 250}
    -0.000617 (0.000187) with: {'learning_rate': 3, 'n_estimators': 300}
    -0.000615 (0.000190) with: {'learning_rate': 3, 'n_estimators': 350}
    -0.000615 (0.000192) with: {'learning_rate': 3, 'n_estimators': 400}


.. code:: ipython3

    # 12. Grid search : KerasNNRegressor
    '''
    nn_shape : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    '''
    #Add Deep Learning Regressor
    if ( EnableDeepLearningRegreesorFlag == 1):
        param_grid={'nn_shape': [(20,), (50,), (20,20), (20, 30, 20)]}
        model = KerasNNRegressor()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_train, Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


 ### 6.2. Grid Search- Time Series Models

.. code:: ipython3

    #Grid Search for ARIMA Model
    #Change p,d and q and check for the best result

    # evaluate an ARIMA model for a given order (p,d,q)
    #Assuming that the train and Test Data is already defined before
    def evaluate_arima_model(arima_order):
        #predicted = list()
        modelARIMA=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=arima_order)
        model_fit = modelARIMA.fit()
        #error on the test set
    #     tr_len = len(X_train_ARIMA)
    #     to_len = len(X_train_ARIMA) + len(X_validation_ARIMA)
    #     predicted = model_fit.predict(start = tr_len -1 ,end = to_len -1, exog = X_validation_ARIMA)[1:]
    #     error = mean_squared_error(predicted, Y_validation)
        # error on the training set
        error = mean_squared_error(Y_train, model_fit.fittedvalues)
        return error

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(p_values, d_values, q_values):
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = evaluate_arima_model(order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.7f' % (order,mse))
                    except:
                        continue
        print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))

    # evaluate parameters
    p_values = [0, 1, 2]
    d_values = range(0, 2)
    q_values = range(0, 2)
    warnings.filterwarnings("ignore")
    evaluate_models(p_values, d_values, q_values)


.. parsed-literal::

    ARIMA(0, 0, 0) MSE=0.0008313
    ARIMA(0, 0, 1) MSE=0.0006774
    ARIMA(1, 0, 0) MSE=0.0004115
    ARIMA(1, 0, 1) MSE=0.0004115
    ARIMA(2, 0, 0) MSE=0.0004115
    ARIMA(2, 0, 1) MSE=0.0004089
    Best ARIMA(2, 0, 1) MSE=0.0004089


.. code:: ipython3

    #Grid Search for LSTM Model

    # evaluate an LSTM model for a given order (p,d,q)
    def evaluate_LSTM_model(neurons=12, learn_rate = 0.01, momentum=0):
        #predicted = list()
        LSTMModel = create_LSTMmodel(neurons, learn_rate, momentum)
        LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM,epochs=50, batch_size=72, verbose=0, shuffle=False)
        predicted = LSTMModel.predict(X_validation_LSTM)
        error = mean_squared_error(predicted, Y_validation)
        return error

    # evaluate combinations of different variables of LSTM Model
    def evaluate_combinations_LSTM(neurons, learn_rate, momentum):
        best_score, best_cfg = float("inf"), None
        for n in neurons:
            for l in learn_rate:
                for m in momentum:
                    combination = (n,l,m)
                    try:
                        mse = evaluate_LSTM_model(n,l,m)
                        if mse < best_score:
                            best_score, best_cfg = mse, combination
                        print('LSTM%s MSE=%.7f' % (combination,mse))
                    except:
                        continue
        print('Best LSTM%s MSE=%.7f' % (best_cfg, best_score))

    # evaluate parameters
    neurons = [1, 5]
    learn_rate = [0.001, 0.3]
    momentum = [0.0, 0.9]
    #Other Parameters can be modified as well
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    warnings.filterwarnings("ignore")
    evaluate_combinations_LSTM(neurons,learn_rate,momentum)


.. parsed-literal::

    LSTM(1, 0.001, 0.0) MSE=0.0009191
    LSTM(1, 0.001, 0.9) MSE=0.0009221
    LSTM(1, 0.3, 0.0) MSE=0.0009202
    LSTM(1, 0.3, 0.9) MSE=0.0009252
    LSTM(5, 0.001, 0.0) MSE=0.0009294
    LSTM(5, 0.001, 0.9) MSE=0.0009371
    LSTM(5, 0.3, 0.0) MSE=0.0008902
    LSTM(5, 0.3, 0.9) MSE=0.0009274
    Best LSTM(5, 0.3, 0.0) MSE=0.0008902


 # 7. Finalise the Model

Let us select one of the model to finalize the data. Looking at the
results for the Random Forest Model. Looking at the results for the
RandomForestRegressor model

 ## 7.1. Results on the Test Dataset

.. code:: ipython3

    # prepare model
    #scaler = StandardScaler().fit(X_train)
    #rescaledX = scaler.transform(X_train)
    model = RandomForestRegressor(n_estimators=250) # rbf is default kernel
    model.fit(X_train, Y_train)




.. parsed-literal::

    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=250, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)



.. code:: ipython3

    # estimate accuracy on validation set
    # transform the validation dataset
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    #rescaledValidationX = scaler.transform(X_validation)
    predictions = model.predict(X_validation)
    print(mean_squared_error(Y_validation, predictions))
    print(r2_score(Y_validation, predictions))


.. parsed-literal::

    0.0010988744547435773
    0.770991173511421


.. code:: ipython3

    predictions




.. parsed-literal::

    array([-0.0167129 , -0.00438854, -0.01720166, -0.01160175,  0.02395749,
            0.04436422,  0.06058143,  0.05072532,  0.03095209,  0.01356325,
            0.0099564 , -0.00842381,  0.01126788,  0.02380102,  0.03248117,
            0.03365514,  0.04211669,  0.03820695,  0.04439574,  0.0403192 ,
            0.05855186,  0.0601968 ,  0.05554876,  0.05474416,  0.04013627,
            0.03683666,  0.02488511,  0.03476343,  0.03237849,  0.03032998,
            0.03901389,  0.03749494,  0.03193098,  0.03607884, -0.03805946,
           -0.05936762, -0.03391639, -0.04593652, -0.01443007, -0.02085788,
           -0.03487344, -0.03638463, -0.03563088, -0.05473057, -0.11883956,
           -0.04365186, -0.05839629, -0.08177799, -0.10658929, -0.0723141 ,
           -0.06093034, -0.03757991, -0.00596715,  0.04780189,  0.02360587,
           -0.01332828, -0.00785864, -0.03248047, -0.02820984,  0.00942003,
           -0.01978709, -0.0491296 , -0.01526217,  0.02559211, -0.01487458,
            0.01762112,  0.06623189,  0.05028198,  0.03630367,  0.05250258,
            0.02850159,  0.02790157, -0.02360305, -0.01868513,  0.00697606,
           -0.00318529, -0.01077579, -0.02886531, -0.02580073, -0.03335841,
           -0.02376638, -0.0748171 , -0.02263548, -0.04121525, -0.06405095,
           -0.07891746, -0.08067931, -0.11992439, -0.08315298, -0.06357971,
           -0.03311136,  0.00101853, -0.02848928, -0.05496003, -0.04974282,
           -0.00102415,  0.03035821,  0.02785181,  0.02478636,  0.04806699,
            0.04101064,  0.01829691,  0.05229777,  0.02267048,  0.01806651,
            0.0481161 ,  0.03338871,  0.03011497,  0.0037035 ,  0.01198423,
            0.03953344,  0.01174706,  0.03181125,  0.02311203, -0.00352524,
            0.04403766,  0.04774884,  0.06654194,  0.0939478 ,  0.06582213,
            0.06795722,  0.06044436,  0.0311404 ,  0.03947465,  0.06407764,
            0.06366104,  0.08626386,  0.05829807,  0.09237167,  0.06248199,
            0.0558632 ,  0.05059528,  0.05819086,  0.04331141,  0.04031494,
            0.0513537 ,  0.06387019,  0.0623841 ,  0.06447874,  0.05932441,
            0.06000217,  0.06130364,  0.06089483,  0.05959214,  0.05165794,
            0.05256753,  0.04758325,  0.03122795,  0.04806537,  0.06412759,
            0.0575488 ,  0.07847543,  0.09702115,  0.09410662,  0.09518771,
            0.07163633,  0.06018362,  0.07542479,  0.0671117 ,  0.04368917,
            0.04781752,  0.04979013,  0.05813182,  0.02939386,  0.03673248,
            0.04364695,  0.06457493,  0.04641177,  0.03707081,  0.03067347,
            0.02349352,  0.01301813, -0.01934682, -0.02268155, -0.02533962,
           -0.03180231, -0.03166371, -0.04222748, -0.03458348, -0.01020102,
            0.02253789,  0.04601804,  0.04959232,  0.04032586,  0.05536184,
            0.05111265,  0.04350028,  0.03341034,  0.04390949,  0.06987675,
            0.0644903 ,  0.07425782,  0.05582975,  0.05744612,  0.07294485,
            0.0579009 ,  0.08598981,  0.10871372,  0.1015554 ,  0.07215359,
            0.05582903,  0.03559519,  0.03959765,  0.03844638,  0.05848534,
            0.0477716 ,  0.03641636,  0.0372103 ,  0.01364644,  0.01196397,
            0.01210995,  0.01321671,  0.01675325,  0.04413029,  0.04326638,
            0.04270718,  0.00684118,  0.01085247, -0.00255595, -0.02286433,
           -0.0147629 , -0.00803585,  0.01157987, -0.00253967, -0.01876178,
           -0.00701802, -0.01346794, -0.01603001,  0.00978598,  0.01435514,
           -0.00074282,  0.00544204, -0.0142785 , -0.04998315, -0.05158218,
           -0.03088809, -0.02809887,  0.01282912,  0.01988682,  0.03721984,
            0.03890533,  0.03490906, -0.00384669, -0.00531803,  0.02578539,
           -0.00358208,  0.03529414,  0.03129913,  0.0088258 , -0.00174584,
            0.04051517,  0.0025114 ,  0.00020497,  0.03579171,  0.03472466,
            0.0328713 ,  0.00938281,  0.00113179, -0.00632527, -0.0106241 ,
           -0.01208661, -0.0109079 , -0.01034006, -0.01091301,  0.00576056,
            0.02670394,  0.01198276,  0.00566075,  0.03272409,  0.02710886,
            0.00203157, -0.01606232, -0.00986223, -0.01519102, -0.00571757,
            0.00651934,  0.00859579,  0.04311555,  0.02940964,  0.05489656,
            0.04774512,  0.03694486,  0.04302572,  0.05375753,  0.03606795,
            0.04383782,  0.04516068,  0.04173705,  0.04564357,  0.08100234,
            0.09733963,  0.079205  ,  0.10191097,  0.09349076,  0.07248063,
            0.07601309,  0.06137686,  0.06087738,  0.04779072,  0.04437849,
            0.04292305,  0.04440576,  0.04354415,  0.05775496,  0.0422338 ,
            0.03701114,  0.04643909,  0.04832221,  0.02886001,  0.02418378,
            0.03465933,  0.04660973,  0.04862381,  0.04718371,  0.04777968,
            0.04225872,  0.02693193,  0.04210093,  0.08859046,  0.0577928 ,
            0.05424162,  0.04425763,  0.04914569,  0.05827294,  0.07302622,
            0.05615673,  0.05454844,  0.0636011 ,  0.07458665,  0.07352579,
            0.06051135,  0.05680841,  0.04436114,  0.0276868 ,  0.04065472,
            0.07529548,  0.07334085,  0.07420636,  0.12768835,  0.12557607,
            0.15197126,  0.13704117,  0.14095683,  0.13725351,  0.12839587,
            0.1271493 ,  0.13057559,  0.11702096,  0.11071868,  0.08927316,
            0.07623848,  0.02815564,  0.045036  , -0.03619496, -0.02910809,
           -0.01998923, -0.01754365, -0.07310221, -0.08735329, -0.12336669,
           -0.12054098, -0.12389039, -0.10056516, -0.12019732, -0.1006773 ,
           -0.12272242, -0.09908782, -0.10078916, -0.10281921, -0.11932   ,
           -0.12394929, -0.07523005, -0.05682079, -0.01546912, -0.07255817,
           -0.07298098, -0.07628215, -0.06934202,  0.02163013,  0.04409767,
            0.02553263,  0.04722916,  0.17142788,  0.07575054,  0.18675236,
            0.18821385,  0.18214007,  0.1821226 ,  0.18667874,  0.11411365,
            0.16722662,  0.09534586,  0.10113332,  0.08780007,  0.10610973,
            0.10871661,  0.10965673,  0.16483241,  0.08664926,  0.10180859,
            0.10586637,  0.12560925,  0.09032735,  0.04148171,  0.03237559,
            0.03337614,  0.03844717,  0.03321158,  0.09264446,  0.04440193,
            0.07560031,  0.02262409,  0.03612574,  0.07382977,  0.03103831,
            0.02792928,  0.06119375,  0.02706105,  0.00241605,  0.02394367,
            0.02341902,  0.02406397,  0.01848726,  0.03604506,  0.04319433,
            0.05985671,  0.04314911,  0.06628985,  0.03415608,  0.09323744,
            0.09162318,  0.06616782,  0.08581251,  0.0680896 ,  0.07405359,
            0.09212332,  0.0918069 ,  0.09039409,  0.11782611,  0.09661128,
            0.10242397,  0.10471816,  0.07383994,  0.08950468,  0.08463576,
            0.08625374,  0.04592718,  0.04495769,  0.07856623,  0.04030275,
            0.04163912])



 ## 7.2. Variable Intuition/Feature Importance Let us look into the
Feature Importance of the Random Forest model

.. code:: ipython3

    import pandas as pd
    import numpy as np
    model = RandomForestRegressor()
    model.fit(X_train,Y_train)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based regressors
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    pyplot.show()


.. parsed-literal::

    [0.88063619 0.01111496 0.01045559 0.0105939  0.0119909  0.00888741
     0.00891842 0.01207416 0.01051047 0.01155166 0.0117194  0.01154695]



.. image:: output_105_1.png


 ## 7.3. Save Model for Later Use

.. code:: ipython3

    # Save Model Using Pickle
    from pickle import dump
    from pickle import load

    # save the model to disk
    filename = 'finalized_model.sav'
    dump(model, open(filename, 'wb'))

.. code:: ipython3

    # some time later...
    # load the model from disk
    loaded_model = load(open(filename, 'rb'))
    # estimate accuracy on validation set
    #rescaledValidationX = scaler.transform(X_validation) #in case the data is scaled.
    #predictions = model.predict(rescaledValidationX)
    predictions = model.predict(X_validation)
    result = mean_squared_error(Y_validation, predictions)
    print(result)


.. parsed-literal::

    0.0010980621870578236
