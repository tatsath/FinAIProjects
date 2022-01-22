.. _APP_Stock:

Stock Return Prediction
-----------------------

In this case study we will use various supervised learning-based models
to predict the stock price of Microsoft using correlated assets and its
own historical data.

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
   -  `4.2.Feature Selection <#3.2>`__

-  `5.Evaluate Algorithms and Models <#4>`__

   -  `5.1. Train/Test Split <#4.1>`__
   -  `5.2. Evaluation Metrics <#4.2>`__
   -  `5.3. Compare Models and Algorithms <#4.3>`__

      -  `5.3.1 Machine Learning models-scikit-learn <#4.3.1>`__
      -  `5.3.2 Time Series based Models-ARIMA and LSTM <#4.3.2>`__

-  `6. Model Tuning and grid search <#5>`__
-  `7. Finalise the model <#6>`__

   -  `7.1. Result on the test dataset <#6.1>`__
   -  `7.2. Save Model for Later Use <#6.2>`__

 # 1. Problem Definition

In the supervised regression framework used for this case study, weekly
return of the Microsoft stock is the predicted variable. We need to
understand what affects Microsoft stock price and hence incorporate as
much information into the model.

For this case study, other than the historical data of Microsoft, the
independent variables used are the following potentially correlated
assets: \* Stocks: IBM (IBM) and Alphabet (GOOGL) \* Currency: USD/JPY
and GBP/USD \* Indices: S&P 500, Dow Jones and VIX

 # 2. Getting Started- Loading the data and python packages

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

Next, we extract the data required for our analysis using pandas
datareader.

.. code:: ipython3

    stk_tickers = ['MSFT', 'IBM', 'GOOGL']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred')
    idx_data = web.DataReader(idx_tickers, 'fred')

Next, we need a series to predict. We choose to predict using weekly
returns. We approximate this by using 5 business day period returns.

.. code:: ipython3

    return_period = 5

We now define our Y series and our X series

Y: MSFT **Future** Returns

X:

::

   a. GOOGL 5 Business Day Returns
   b. IBM 5 Business DayReturns
   c. USD/JPY 5 Business DayReturns
   d. GBP/USD 5 Business DayReturns
   e. S&P 500 5 Business DayReturns
   f. Dow Jones 5 Business DayReturns
   g. MSFT 5 Business Day Returns
   h. MSFT 15 Business Day Returns
   i. MSFT 30 Business Day Returns
   j. MSFT 60 Business Day Returns

We remove the MSFT past returns when we use the Time series models.

.. code:: ipython3

    Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1]+'_pred'

    X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X4 = pd.concat([np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(i) for i in [return_period, return_period*3, return_period*6, return_period*12]], axis=1).dropna()
    X4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']

    X = pd.concat([X1, X2, X3, X4], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]

 # 3. Exploratory Data Analysis

 ## 3.1. Descriptive Statistics

Lets have a look at the dataset we have

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
          <th>MSFT_pred</th>
          <th>GOOGL</th>
          <th>IBM</th>
          <th>DEXJPUS</th>
          <th>DEXUSUK</th>
          <th>SP500</th>
          <th>DJIA</th>
          <th>VIXCLS</th>
          <th>MSFT_DT</th>
          <th>MSFT_3DT</th>
          <th>MSFT_6DT</th>
          <th>MSFT_12DT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>465.000</td>
          <td>465.000</td>
          <td>4.650e+02</td>
          <td>4.650e+02</td>
          <td>4.650e+02</td>
          <td>465.000</td>
          <td>465.000</td>
          <td>465.000</td>
          <td>465.000</td>
          <td>465.000</td>
          <td>465.000</td>
          <td>465.000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>0.004</td>
          <td>0.002</td>
          <td>1.536e-04</td>
          <td>3.163e-04</td>
          <td>-3.611e-04</td>
          <td>0.001</td>
          <td>0.001</td>
          <td>0.003</td>
          <td>0.003</td>
          <td>0.011</td>
          <td>0.023</td>
          <td>0.044</td>
        </tr>
        <tr>
          <th>std</th>
          <td>0.029</td>
          <td>0.033</td>
          <td>2.601e-02</td>
          <td>1.321e-02</td>
          <td>1.256e-02</td>
          <td>0.018</td>
          <td>0.017</td>
          <td>0.151</td>
          <td>0.029</td>
          <td>0.047</td>
          <td>0.065</td>
          <td>0.088</td>
        </tr>
        <tr>
          <th>min</th>
          <td>-0.122</td>
          <td>-0.138</td>
          <td>-1.011e-01</td>
          <td>-5.245e-02</td>
          <td>-1.112e-01</td>
          <td>-0.075</td>
          <td>-0.071</td>
          <td>-0.496</td>
          <td>-0.122</td>
          <td>-0.137</td>
          <td>-0.212</td>
          <td>-0.223</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>-0.014</td>
          <td>-0.015</td>
          <td>-1.326e-02</td>
          <td>-7.166e-03</td>
          <td>-7.748e-03</td>
          <td>-0.007</td>
          <td>-0.008</td>
          <td>-0.093</td>
          <td>-0.014</td>
          <td>-0.016</td>
          <td>-0.013</td>
          <td>-0.015</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>0.004</td>
          <td>0.004</td>
          <td>1.414e-03</td>
          <td>0.000e+00</td>
          <td>-5.608e-04</td>
          <td>0.004</td>
          <td>0.003</td>
          <td>-0.005</td>
          <td>0.004</td>
          <td>0.015</td>
          <td>0.026</td>
          <td>0.056</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>0.021</td>
          <td>0.021</td>
          <td>1.579e-02</td>
          <td>7.538e-03</td>
          <td>7.598e-03</td>
          <td>0.012</td>
          <td>0.012</td>
          <td>0.081</td>
          <td>0.020</td>
          <td>0.040</td>
          <td>0.062</td>
          <td>0.100</td>
        </tr>
        <tr>
          <th>max</th>
          <td>0.113</td>
          <td>0.230</td>
          <td>9.713e-02</td>
          <td>5.811e-02</td>
          <td>5.023e-02</td>
          <td>0.055</td>
          <td>0.056</td>
          <td>0.781</td>
          <td>0.113</td>
          <td>0.146</td>
          <td>0.214</td>
          <td>0.266</td>
        </tr>
      </tbody>
    </table>
    </div>



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
          <th>MSFT_pred</th>
          <th>GOOGL</th>
          <th>IBM</th>
          <th>DEXJPUS</th>
          <th>DEXUSUK</th>
          <th>SP500</th>
          <th>DJIA</th>
          <th>VIXCLS</th>
          <th>MSFT_DT</th>
          <th>MSFT_3DT</th>
          <th>MSFT_6DT</th>
          <th>MSFT_12DT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2010-03-31</th>
          <td>0.021</td>
          <td>1.741e-02</td>
          <td>-0.002</td>
          <td>1.630e-02</td>
          <td>0.018</td>
          <td>0.001</td>
          <td>0.002</td>
          <td>0.002</td>
          <td>-0.012</td>
          <td>0.011</td>
          <td>0.024</td>
          <td>-0.050</td>
        </tr>
        <tr>
          <th>2010-04-08</th>
          <td>0.031</td>
          <td>6.522e-04</td>
          <td>-0.005</td>
          <td>-7.166e-03</td>
          <td>-0.001</td>
          <td>0.007</td>
          <td>0.000</td>
          <td>-0.058</td>
          <td>0.021</td>
          <td>0.010</td>
          <td>0.044</td>
          <td>-0.007</td>
        </tr>
        <tr>
          <th>2010-04-16</th>
          <td>0.009</td>
          <td>-2.879e-02</td>
          <td>0.014</td>
          <td>-1.349e-02</td>
          <td>0.002</td>
          <td>-0.002</td>
          <td>0.002</td>
          <td>0.129</td>
          <td>0.011</td>
          <td>0.022</td>
          <td>0.069</td>
          <td>0.007</td>
        </tr>
        <tr>
          <th>2010-04-23</th>
          <td>-0.014</td>
          <td>-9.424e-03</td>
          <td>-0.005</td>
          <td>2.309e-02</td>
          <td>-0.002</td>
          <td>0.021</td>
          <td>0.017</td>
          <td>-0.100</td>
          <td>0.009</td>
          <td>0.060</td>
          <td>0.059</td>
          <td>0.047</td>
        </tr>
        <tr>
          <th>2010-04-30</th>
          <td>-0.079</td>
          <td>-3.604e-02</td>
          <td>-0.008</td>
          <td>6.369e-04</td>
          <td>-0.004</td>
          <td>-0.025</td>
          <td>-0.018</td>
          <td>0.283</td>
          <td>-0.014</td>
          <td>0.007</td>
          <td>0.031</td>
          <td>0.069</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 3.2. Data Visualization

Next, lets look at the distribution of the data over the entire period

.. code:: ipython3

    dataset.hist(bins=50, sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
    pyplot.show()



.. image:: output_23_0.png


The above histogram shows the distribution for each series individually.
Next, lets look at the density distribution over the same x axis scale.

.. code:: ipython3

    dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=True, legend=True, fontsize=1, figsize=(15,15))
    pyplot.show()



.. image:: output_25_0.png


We can see that the vix has a much larger variance compared to the other
distributions.

In order to get a sense of the interdependence of the data we look at
the scatter plot and the correlation matrix

.. code:: ipython3

    correlation = dataset.corr()
    pyplot.figure(figsize=(15,15))
    pyplot.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1d023aa0f98>




.. image:: output_28_1.png


Looking at the correlation plot above, we see some correlation of the
predicted vari‐ able with the lagged 5 days, 15days, 30 days and 60 days
return of MSFT.

.. code:: ipython3

    pyplot.figure(figsize=(15,15))
    scatter_matrix(dataset,figsize=(12,12))
    pyplot.show()



.. parsed-literal::

    <Figure size 1080x1080 with 0 Axes>



.. image:: output_30_1.png


Looking at the scatter plot above, we see some linear relationship of
the predicted variable the lagged 15 days, 30 days and 60 days return of
MSFT.

 ## 3.3. Time Series Analysis

Next, we look at the seasonal decomposition of our time series

.. code:: ipython3

    res = sm.tsa.seasonal_decompose(Y,freq=52)
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    pyplot.show()



.. image:: output_34_0.png


We can see that for MSFT there has been a general trend upwards. This
should show up in our the constant/bias terms in our models

 ## 4. Data Preparation

 ## 4.2. Feature Selection

We use sklearn’s SelectKBest function to get a sense of feature
importance.

.. code:: ipython3

    bestfeatures = SelectKBest(k=5, score_func=f_regression)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    featureScores.nlargest(10,'Score').set_index('Specs')  #print 10 best features




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
          <th>Score</th>
        </tr>
        <tr>
          <th>Specs</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>IBM</th>
          <td>9.639</td>
        </tr>
        <tr>
          <th>MSFT_DT</th>
          <td>5.257</td>
        </tr>
        <tr>
          <th>MSFT_3DT</th>
          <td>4.534</td>
        </tr>
        <tr>
          <th>MSFT_12DT</th>
          <td>4.503</td>
        </tr>
        <tr>
          <th>GOOGL</th>
          <td>3.643</td>
        </tr>
        <tr>
          <th>SP500</th>
          <td>2.964</td>
        </tr>
        <tr>
          <th>DJIA</th>
          <td>2.706</td>
        </tr>
        <tr>
          <th>MSFT_6DT</th>
          <td>2.205</td>
        </tr>
        <tr>
          <th>DEXUSUK</th>
          <td>1.974</td>
        </tr>
        <tr>
          <th>VIXCLS</th>
          <td>1.928</td>
        </tr>
      </tbody>
    </table>
    </div>



We see that IBM seems to be the most important feature and vix being the
least important.

 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split and Evaluation Metrics

Next, we start by splitting our data in training and testing chunks. If
we are going to use Time series models we have to split the data in
continous series.

.. code:: ipython3

    validation_size = 0.2

    #In case the data is not dependent on the time series, then train and test split randomly
    # seed = 7
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    #In case the data is not dependent on the time series, then train and test split should be done based on sequential sample
    #This can be done by selecting an arbitrary split point in the ordered list of observations and creating two new datasets.
    train_size = int(len(X) * (1-validation_size))
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

 ## 5.2. Test Options and Evaluation Metrics

.. code:: ipython3

    num_folds = 10
    seed = 7
    # scikit is moving away from mean_squared_error.
    # In order to avoid confusion, and to allow comparison with other models, we invert the final scores
    scoring = 'neg_mean_squared_error'

 ## 5.3. Compare Models and Algorithms

 ### 5.3.1 Machine Learning models-from scikit-learn

Regression and Tree Regression algorithms
'''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))

Neural Network algorithms
'''''''''''''''''''''''''

.. code:: ipython3

    models.append(('MLP', MLPRegressor()))

Ensable Models
''''''''''''''

.. code:: ipython3

    # Boosting methods
    models.append(('ABR', AdaBoostRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))
    # Bagging methods
    models.append(('RFR', RandomForestRegressor()))
    models.append(('ETR', ExtraTreesRegressor()))

Once we have selected all the models, we loop over each of them. First
we run the K-fold analysis. Next we run the model on the entire training
and testing dataset.

.. code:: ipython3

    names = []
    kfold_results = []
    test_results = []
    train_results = []
    for name, model in models:
        names.append(name)

        ## K Fold analysis:

        kfold = KFold(n_splits=num_folds, random_state=seed)
        #converted mean square error to positive. The lower the beter
        cv_results = -1* cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)


        # Full Training period
        res = model.fit(X_train, Y_train)
        train_result = mean_squared_error(res.predict(X_train), Y_train)
        train_results.append(train_result)

        # Test results
        test_result = mean_squared_error(res.predict(X_test), Y_test)
        test_results.append(test_result)

        msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
        print(msg)


.. parsed-literal::

    LR: 0.000913 (0.000396) 0.000847 0.000610
    LASSO: 0.000882 (0.000380) 0.000881 0.000611
    EN: 0.000882 (0.000380) 0.000881 0.000611
    KNN: 0.001051 (0.000352) 0.000738 0.000838
    CART: 0.002026 (0.000646) 0.000000 0.001211
    SVR: 0.000945 (0.000418) 0.000910 0.000697
    MLP: 0.001084 (0.000414) 0.001159 0.000860
    ABR: 0.001036 (0.000419) 0.000640 0.000737
    GBR: 0.001123 (0.000493) 0.000240 0.000690
    RFR: 0.001087 (0.000415) 0.000179 0.000813
    ETR: 0.001143 (0.000391) 0.000000 0.000812


K Fold results
^^^^^^^^^^^^^^

We being by looking at the K Fold results

.. code:: ipython3

    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison: Kfold results')
    ax = fig.add_subplot(111)
    pyplot.boxplot(kfold_results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15,8)
    pyplot.show()



.. image:: output_59_0.png


We see the linear regression and the regularized regression including
the Lasso regression (LASSO) and elastic net (EN) seem to do a good job.

Training and Test error
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()

    ind = np.arange(len(names))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.bar(ind - width/2, train_results,  width=width, label='Train Error')
    pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
    fig.set_size_inches(15,8)
    pyplot.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    pyplot.show()



.. image:: output_62_0.png


Looking at the training and test error, we still see a better
performance of the linear models. Some of the algorithms, such as the
decision tree regressor (CART) overfit on the training data and produced
very high error on the test set and these models should be avoided.
Ensemble models, such as gradient boosting regression (GBR) and random
forest regression (RFR) have low bias but high variance. We also see
that the artificial neural network (shown as MLP is the chart) algorithm
shows higher errors both in training set and test set, which is perhaps
due to the linear relationship of the variables not captured accurately
by ANN or improper hyperparameters or insuffi‐ cient training of the
model.

 ### 5.3.1 Time Series based models-ARIMA and LSTM

Let us first prepare the dataset for ARIMA models, by having only the
correlated varriables as exogenous variables.

Time Series Model - ARIMA Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    X_train_ARIMA=X_train.loc[:, ['GOOGL', 'IBM', 'DEXJPUS', 'SP500', 'DJIA', 'VIXCLS']]
    X_test_ARIMA=X_test.loc[:, ['GOOGL', 'IBM', 'DEXJPUS', 'SP500', 'DJIA', 'VIXCLS']]
    tr_len = len(X_train_ARIMA)
    te_len = len(X_test_ARIMA)
    to_len = len (X)

.. code:: ipython3

    modelARIMA=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=[1,0,0])
    model_fit = modelARIMA.fit()

.. code:: ipython3

    error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
    predicted = model_fit.predict(start = tr_len -1 ,end = to_len -1, exog = X_test_ARIMA)[1:]
    error_Test_ARIMA = mean_squared_error(Y_test,predicted)
    error_Test_ARIMA




.. parsed-literal::

    0.0005931919240399084



LSTM Model
~~~~~~~~~~

.. code:: ipython3

    seq_len = 2 #Length of the seq for the LSTM

    Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len-1:], np.array(Y_test)
    X_train_LSTM = np.zeros((X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]))
    X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
    for i in range(seq_len):
        X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
        X_test_LSTM[:, i, :] = np.array(X)[X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]


.. code:: ipython3

    # Lstm Network
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
    LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_test_LSTM, Y_test_LSTM),epochs=330, batch_size=72, verbose=0, shuffle=False)

.. code:: ipython3

    #Visual plot to check if the error is reducing
    pyplot.plot(LSTMModel_fit.history['loss'], label='train')
    pyplot.plot(LSTMModel_fit.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()



.. image:: output_73_0.png


.. code:: ipython3

    error_Training_LSTM = mean_squared_error(Y_train_LSTM, LSTMModel.predict(X_train_LSTM))
    predicted = LSTMModel.predict(X_test_LSTM)
    error_Test_LSTM = mean_squared_error(Y_test,predicted)

Append to previous results
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    test_results.append(error_Test_ARIMA)
    test_results.append(error_Test_LSTM)

    train_results.append(error_Training_ARIMA)
    train_results.append(error_Training_LSTM)

    names.append("ARIMA")
    names.append("LSTM")

Overall Comparison of all the algorithms ( including Time Series Algorithms)
----------------------------------------------------------------------------

.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()

    ind = np.arange(len(names))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig.suptitle('Comparing the performance of various algorthims on the Train and Test Dataset')
    ax = fig.add_subplot(111)
    pyplot.bar(ind - width/2, train_results,  width=width, label='Train Error')
    pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
    fig.set_size_inches(15,8)
    pyplot.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    pyplot.ylabel('Mean Square Error')
    pyplot.show()



.. image:: output_78_0.png


Looking at the chart above, we find time series based ARIMA model
comparable to the linear supervised-regression models such as Linear
Regression (LR), Lasso Regres‐ sion (LASSO) and Elastic Net (EN). This
can primarily be due to the strong linear relationship as discussed
before. The LSTM model performs decently, however, ARIMA model
outperforms the LSTM model in the test set. Hence, we select the ARIMA
model for the model tuning.

 # 6. Model Tuning and Grid Search

As shown in the chart above the ARIMA model is one of the best mode, so
we perform the model tuning of the ARIMA model. The default order of
ARIMA model is [1,0,0]. We perform a grid search with different
combination p,d and q in the ARIMA model’s order.

.. code:: ipython3

    #Grid Search for ARIMA Model
    #Change p,d and q and check for the best result

    # evaluate an ARIMA model for a given order (p,d,q)
    #Assuming that the train and Test Data is already defined before
    def evaluate_arima_model(arima_order):
        #predicted = list()
        modelARIMA=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=arima_order)
        model_fit = modelARIMA.fit()
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

    ARIMA(0, 0, 0) MSE=0.0008632
    ARIMA(0, 0, 1) MSE=0.0008589
    ARIMA(1, 0, 0) MSE=0.0008592
    ARIMA(1, 0, 1) MSE=0.0008485
    ARIMA(2, 0, 0) MSE=0.0008586
    ARIMA(2, 0, 1) MSE=0.0008481
    Best ARIMA(2, 0, 1) MSE=0.0008481


 # 7. Finalise the Model

 ## 7.1. Results on the Test Dataset

.. code:: ipython3

    # prepare model
    modelARIMA_tuned=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=[2,0,1])
    model_fit_tuned = modelARIMA_tuned.fit()

.. code:: ipython3

    # estimate accuracy on validation set
    predicted_tuned = model_fit.predict(start = tr_len -1 ,end = to_len -1, exog = X_test_ARIMA)[1:]
    print(mean_squared_error(Y_test,predicted_tuned))


.. parsed-literal::

    0.0005970582461404503


After tuning the model and picking the best ARIMA model or the order 2,0
and 1 we select this model and can it can be used for the modeling
purpose.

 ## 7.2. Save Model for Later Use

.. code:: ipython3

    # Save Model Using Pickle
    from pickle import dump
    from pickle import load

    # save the model to disk
    filename = 'finalized_model.sav'
    dump(model_fit_tuned, open(filename, 'wb'))

.. code:: ipython3

    #Use the following code to produce the comparison of actual vs. predicted
    # predicted_tuned.index = Y_test.index
    # pyplot.plot(np.exp(Y_test).cumprod(), 'r') # plotting t, a separately
    # pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b')
    # pyplot.rcParams["figure.figsize"] = (8,5)
    # pyplot.show()

Summary
~~~~~~~

We can conclude that simple models - linear regression, regularized
regression ( i.e. Lasso and elastic net) - along with the time series
model such as ARIMA are promis‐ ing modelling approaches for asset price
prediction problem. These models can enable financial practitioners to
model time dependencies with a very flexible approach. The overall
approach presented in this case study may help us encounter overfitting
and underfitting which are some of the key challenges in the prediction
problem in finance. We should also note that we can use better set of
indicators, such as P/E ratio, trading volume, technical indicators or
news data, which might lead to better results. We will demonstrate this
in some of the case studies in the book. Overall, we created a
supervised-regression and time series modelling framework which allows
us to perform asset class prediction using historical data to generate
results and analyze risk and profitability before risking any actual
capital.
