.. _Port_Clustering:


In this case study, we will implement hierarchical risk parity based on
clustering methods and compare it against other portfolio allocation
methods. The case study is based on the paper *“Building Diversified
Portfolios that Outperform Out-ofSample” (2016) by Marcos Lopez de
Prado.*

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

   -  `5.1. Building Hierarchy Graph/ Dendogramg <#4.1>`__
   -  `5.2. Steps for Hierarchial Risk Parity <#4.2>`__

-  `6.Backtesting-Out Of Sample <#5>`__

 # 1. Problem Definition

Our goal in this case study is to maximize risk-adjusted returns using a
clustering reduction-based algorithm on a dataset of stocks to allocate
capital into different asset classes.

 # 2. Getting Started- Loading the data and python packages

 ## 2.1. Loading the python packages

Checking is the additional packages needed are present, if not install
them. Let us chek is the cvxopt package is present, if not install it.
This package is checked separately as it is not included in
requirement.txt of this book repository as the package is not used
across any other case study of thie book.

.. code:: ipython3

    import pkg_resources
    import pip
    installedPackages = {pkg.key for pkg in pkg_resources.working_set}
    if 'cvxopt' not in installedPackages :
        !pip install cvxopt==1.2.5

.. code:: ipython3

    # Load libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas import read_csv, set_option
    from pandas.plotting import scatter_matrix
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from datetime import date

    #Import Model Packages
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn import cluster, covariance, manifold

    #Package for optimization of mean variance optimization
    import cvxopt as opt
    from cvxopt import blas, solvers


 ## 2.2. Loading the Data

.. code:: ipython3

    # load dataset
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



 ## 3.2. Data Visualization

We will look at the data visualisation in the later sections.

 ## 4. Data Preparation

 ## 4.1. Data Cleaning We check for the NAs in the rows, either drop
them or fill them with the mean of the column.

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())
    dataset.shape


.. parsed-literal::

    Null Values = True




.. parsed-literal::

    (448, 502)



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

For the purpose of clustering, we will be using annual returns.
Additionally, we will train the data followed by testing. Let us prepare
the dataset for training and testing, by separating 20% of the dataset
for testing followed by generating the return series.

.. code:: ipython3

    X= dataset.copy('deep')
    row= len(X)
    train_len = int(row*.8)

.. code:: ipython3

    X_train = X.head(train_len)
    X_test = X.tail(row-train_len)

.. code:: ipython3

    #Calculate percentage return
    returns = X_train.pct_change().dropna()
    returns_test=X_test.pct_change().dropna()

The parameters to clusters are the indices and the variables used in the
clustering are the columns. Hence the data is in the right format to be
fed to the clustering algorithms

 # 5. Evaluate Algorithms and Models

 ## 5.1. Building Hierarchy Graph/ Dendogram

The first step is to look for clusters of corre‐ lations using the
agglomerate hierarchical clustering technique. The hierarchy class has a
dendrogram method which takes the value returned by the linkage method
of the same class.

Linkage does the actual clustering in one line of code, and returns a
list of the clusters joined in the format: Z=[stock_1, stock_2,
distance, sample_count]

Compute Correlation
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    def correlDist(corr):
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.)**.5  # distance matrix
        return dist

.. code:: ipython3

    #Calulate linkage
    dist = correlDist(returns.corr())
    link = linkage(dist, 'ward')
    link[0]




.. parsed-literal::

    array([24.        , 25.        ,  0.09110116,  2.        ])



Computation of linkages is followed by the visualization of the clusters
through a dendrogram, which displays a cluster tree. The leaves are the
individual stocks and the root is the final single cluster. The
“distance” between each cluster is shown on the y-axis, and thus the
longer the branches are, the less correlated two clusters are.

.. code:: ipython3

    #Plot Dendogram
    plt.figure(figsize=(20, 7))
    plt.title("Dendrograms")
    dendrogram(link,labels = X.columns)
    plt.show()



.. image:: output_40_0.png


 ## 5.2 Steps for Hierarchial Risk Parity

Quasi-diagonalization and getting the weights for Hierarchial Risk Parity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ‘quasi-diagonalization’ is a process usually known as matrix seriation
and which can be performed using hierarchical clustering. This process
reorganize the covariance matrix so similar investments will be placed
together. This matrix diagonalization allow us to distribute weights
optimally following an inverse-variance allocation.

.. code:: ipython3

    def getQuasiDiag(link):
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

Recursive bisection
^^^^^^^^^^^^^^^^^^^

This step distributes the allocation through recursive bisection based
on cluster covariance.

.. code:: ipython3

    def getClusterVar(cov,cItems):
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems] # matrix slice
        w_=getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar



    def getRecBipart(cov, sortIx):
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = getClusterVar(cov, cItems0)
                cVar1 = getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w

Comparison against other asset allocation methods:.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main premise of this case study was to develop an alternative to
Markowitz’s Minimum-Variance Portfolio based asset allocation. So, in
this step, we define a functions to compare the performance of the
following asset allocation methods.

::

   1. MVP - Markowitz’s Minimum-Variance Portfolio
   2. HRP - Hierarchial Risk Parity

.. code:: ipython3

    def getMVP(cov):

        cov = cov.T.values
        n = len(cov)
        N = 100
        mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

        # Convert to cvxopt matrices
        S = opt.matrix(cov)
        #pbar = opt.matrix(np.mean(returns, axis=1))
        pbar = opt.matrix(np.ones(cov.shape[0]))

        # Create constraint matrices
        G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)


        # Calculate efficient frontier weights using quadratic programming
        solvers.options['show_progress'] = False
        portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                      for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

        return list(wt)

    def getIVP(cov, **kargs):
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def getHRP(cov, corr):
        # Construct a hierarchical portfolio
        dist = correlDist(corr)
        link = sch.linkage(dist, 'single')
        #plt.figure(figsize=(20, 10))
        #dn = sch.dendrogram(link, labels=cov.index.values)
        #plt.show()
        sortIx = getQuasiDiag(link)
        sortIx = corr.index[sortIx].tolist()
        hrp = getRecBipart(cov, sortIx)
        return hrp.sort_index()

Step 4: Getting the portfolio weights for all types of asset allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    def get_all_portfolios(returns):

        cov, corr = returns.cov(), returns.corr()
        hrp = getHRP(cov, corr)
        mvp = getMVP(cov)
        mvp = pd.Series(mvp, index=cov.index)
        portfolios = pd.DataFrame([mvp, hrp], index=['MVP', 'HRP']).T
        #portfolios = pd.DataFrame([ivp, hrp], index=['IVP', 'HRP']).T
        return portfolios

.. code:: ipython3

    portfolios = get_all_portfolios(returns)

.. code:: ipython3

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,20))
    ax1.pie(portfolios.iloc[:,0], );
    ax1.set_title('MVP',fontsize = 30)
    ax2.pie(portfolios.iloc[:,1]);
    ax2.set_title('HRP',fontsize = 30)
    #portfolios.plot.pie(subplots=True, figsize=(20, 10),legend = False);




.. parsed-literal::

    Text(0.5, 1.0, 'HRP')




.. image:: output_54_1.png


The first pie chart shown the asset allocation of MVP, followed by IVP
and HRP. We clearly see more diversification in HRP as compared to MVP.
On the other hand, IVP evenly spreads weights through all assets, and it
looks similar to HRP. However, this method ignores the correlation
structure between the instruments unlike HRP is included just for
comparison purpose. Let us look at the backtesting results.

 # 6. Backtesting-Out Of Sample

.. code:: ipython3

    Insample_Result=pd.DataFrame(np.dot(returns,np.array(portfolios)), \
                                 columns=['MVP', 'HRP'], index = returns.index)
    OutOfSample_Result=pd.DataFrame(np.dot(returns_test,np.array(portfolios)), \
                                    columns=['MVP', 'HRP'], index = returns_test.index)

.. code:: ipython3

    Insample_Result.cumsum().plot(figsize=(10, 5), title ="In-Sample Results")
    OutOfSample_Result.cumsum().plot(figsize=(10, 5), title ="Out Of Sample Results")




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x142248842b0>




.. image:: output_58_1.png



.. image:: output_58_2.png


In Sample and Out of Sample Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #In_sample Results
    stddev = Insample_Result.std() * np.sqrt(252)
    sharp_ratio = (Insample_Result.mean()*np.sqrt(252))/(Insample_Result).std()
    Results = pd.DataFrame(dict(stdev=stddev, sharp_ratio = sharp_ratio))
    Results




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
          <th>stdev</th>
          <th>sharp_ratio</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MVP</th>
          <td>0.085516</td>
          <td>0.785019</td>
        </tr>
        <tr>
          <th>HRP</th>
          <td>0.126944</td>
          <td>0.523599</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Outof_sample Results
    stddev_oos = OutOfSample_Result.std() * np.sqrt(252)
    sharp_ratio_oos = (OutOfSample_Result.mean()*np.sqrt(252))/(OutOfSample_Result).std()
    Results_oos = pd.DataFrame(dict(stdev_oos=stddev_oos, sharp_ratio_oos = sharp_ratio_oos))
    Results_oos




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
          <th>stdev_oos</th>
          <th>sharp_ratio_oos</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MVP</th>
          <td>0.102761</td>
          <td>0.786621</td>
        </tr>
        <tr>
          <th>HRP</th>
          <td>0.125610</td>
          <td>0.836159</td>
        </tr>
      </tbody>
    </table>
    </div>



Although the in-sample result of MVP look promising, the out of sample
sharp ratio and overall return of portfolio constructed using
hierarchical clustering approach are better. The diversification that
HRP achieves across uncorrelated assets makes the methodology more
robust against shocks

**Conclusion**

Markowitz’s minimum-variance based portfolio allocation is less diverse
and is concentrated in a few stocks. In the hierarchical
clustering-based allocation, the allocation is more diverse and
distributed across many assets. These portfolios then offer better tail
risk management.

Finally, we looked at the backtesting framework, which enables us to
compute and analyze the in-sample and out of sample return and sharpe
ratio of each hypothetical portfolio. This helps in determining the best
performer out of all the portfolio allocation methods. The hierarchical
clustering approaches yielded the best out-of-sample results.
