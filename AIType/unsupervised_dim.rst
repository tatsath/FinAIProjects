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

1. Introduction
------------------------------------------------

We will look at the following models and the related concepts 1.
Principal Component Analysis (PCA) 2. Kernel PCA (KPCA) 3. t-distributed
Stochastic Neighbor Embedding (t-SNE)

2. Getting Started- Loading the data and python packages
------------------------------------------------

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



3. Exploratory Data Analysis
------------------------------------------------

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


.. code:: ipython3

    # types
    set_option('display.max_rows', 500)
    dataset.dtypes


.. code:: ipython3

    # describe data
    set_option('precision', 3)
    dataset.describe()

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


4. Data Preparation
------------------------------------------------

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


5. Evaluate Algorithms and Models
------------------------------------------------

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
