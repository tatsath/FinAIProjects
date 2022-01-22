.. _Class_Fraud:


Fraud Detection
===============

The goal of this case study is to use various classification-based
models to detect whether a transaction is a normal payment or a fraud.

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
   -  `5.2. Evaluate Models <#4.2>`__

-  `6. Model Tuning <#5>`__

   -  `5.1. Model Tuning-Right Evaluation Metric <#5.1>`__
   -  `5.2. Model Tuning-Balancing the sample <#5.2>`__

 # 1. Problem Definition

In the classification framework defined for this case study, the
response variable takes a value of 1 in case the given transaction is
fraud and 0 otherwise.

The datasets contains transactions made by credit cards in September
2013 by european cardholders. This dataset presents transactions that
occurred in two days, where we have 492 frauds out of 284,807
transactions. The dataset is highly unbalanced, the positive class
(frauds) account for 0.172% of all transactions.The task is to get
forecast the fraud. Feature ‘Class’ is the response variable and it
takes value 1 in case of fraud and 0 otherwise.The features are the
result of PCA transformation and aren’t intuitive as far as their names
are concerned.

The data can be downloaded from:
https://www.kaggle.com/mlg-ulb/creditcardfraud

 # 2. Getting Started- Loading the data and python packages

 ## 2.1. Loading the python packages

.. code:: ipython3

    # Load libraries
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot
    from pandas import read_csv, set_option
    from pandas.plotting import scatter_matrix
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    #Libraries for Deep Learning Models
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.optimizers import SGD

    #Libraries for Saving the Model
    from pickle import dump
    from pickle import load


.. parsed-literal::

    Using TensorFlow backend.


 ## 2.2. Loading the Data

We load the data in this step.

Note : Due to limit in the github for the data size, a sample of the data has been loaded in the jupyter notebook repository of this book. However, all the subsequent results in this jupyter notebook is with actual data (144MB) under https://www.kaggle.com/mlg-ulb/creditcardfraud. You should load the full data in case you want to reproduce the results.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # load dataset
    dataset = read_csv('creditcard_sample.csv')
    #dataset = read_csv('creditcard.csv') #Load this for the actual data.

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

    (284807, 31)



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
          <th>Time</th>
          <th>V1</th>
          <th>V2</th>
          <th>V3</th>
          <th>V4</th>
          <th>V5</th>
          <th>V6</th>
          <th>V7</th>
          <th>V8</th>
          <th>V9</th>
          <th>...</th>
          <th>V21</th>
          <th>V22</th>
          <th>V23</th>
          <th>V24</th>
          <th>V25</th>
          <th>V26</th>
          <th>V27</th>
          <th>V28</th>
          <th>Amount</th>
          <th>Class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.0</td>
          <td>-1.360</td>
          <td>-0.073</td>
          <td>2.536</td>
          <td>1.378</td>
          <td>-0.338</td>
          <td>0.462</td>
          <td>0.240</td>
          <td>0.099</td>
          <td>0.364</td>
          <td>...</td>
          <td>-0.018</td>
          <td>0.278</td>
          <td>-0.110</td>
          <td>0.067</td>
          <td>0.129</td>
          <td>-0.189</td>
          <td>0.134</td>
          <td>-0.021</td>
          <td>149.62</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.0</td>
          <td>1.192</td>
          <td>0.266</td>
          <td>0.166</td>
          <td>0.448</td>
          <td>0.060</td>
          <td>-0.082</td>
          <td>-0.079</td>
          <td>0.085</td>
          <td>-0.255</td>
          <td>...</td>
          <td>-0.226</td>
          <td>-0.639</td>
          <td>0.101</td>
          <td>-0.340</td>
          <td>0.167</td>
          <td>0.126</td>
          <td>-0.009</td>
          <td>0.015</td>
          <td>2.69</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.0</td>
          <td>-1.358</td>
          <td>-1.340</td>
          <td>1.773</td>
          <td>0.380</td>
          <td>-0.503</td>
          <td>1.800</td>
          <td>0.791</td>
          <td>0.248</td>
          <td>-1.515</td>
          <td>...</td>
          <td>0.248</td>
          <td>0.772</td>
          <td>0.909</td>
          <td>-0.689</td>
          <td>-0.328</td>
          <td>-0.139</td>
          <td>-0.055</td>
          <td>-0.060</td>
          <td>378.66</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.0</td>
          <td>-0.966</td>
          <td>-0.185</td>
          <td>1.793</td>
          <td>-0.863</td>
          <td>-0.010</td>
          <td>1.247</td>
          <td>0.238</td>
          <td>0.377</td>
          <td>-1.387</td>
          <td>...</td>
          <td>-0.108</td>
          <td>0.005</td>
          <td>-0.190</td>
          <td>-1.176</td>
          <td>0.647</td>
          <td>-0.222</td>
          <td>0.063</td>
          <td>0.061</td>
          <td>123.50</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.0</td>
          <td>-1.158</td>
          <td>0.878</td>
          <td>1.549</td>
          <td>0.403</td>
          <td>-0.407</td>
          <td>0.096</td>
          <td>0.593</td>
          <td>-0.271</td>
          <td>0.818</td>
          <td>...</td>
          <td>-0.009</td>
          <td>0.798</td>
          <td>-0.137</td>
          <td>0.141</td>
          <td>-0.206</td>
          <td>0.502</td>
          <td>0.219</td>
          <td>0.215</td>
          <td>69.99</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 31 columns</p>
    </div>



.. code:: ipython3

    # types
    set_option('display.max_rows', 500)
    dataset.dtypes




.. parsed-literal::

    Time      float64
    V1        float64
    V2        float64
    V3        float64
    V4        float64
    V5        float64
    V6        float64
    V7        float64
    V8        float64
    V9        float64
    V10       float64
    V11       float64
    V12       float64
    V13       float64
    V14       float64
    V15       float64
    V16       float64
    V17       float64
    V18       float64
    V19       float64
    V20       float64
    V21       float64
    V22       float64
    V23       float64
    V24       float64
    V25       float64
    V26       float64
    V27       float64
    V28       float64
    Amount    float64
    Class       int64
    dtype: object



As shown in the results above, the entire data type is float, except
Class which is integer, and the variable names aren’t intuitive.

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
          <th>Time</th>
          <th>V1</th>
          <th>V2</th>
          <th>V3</th>
          <th>V4</th>
          <th>V5</th>
          <th>V6</th>
          <th>V7</th>
          <th>V8</th>
          <th>V9</th>
          <th>...</th>
          <th>V21</th>
          <th>V22</th>
          <th>V23</th>
          <th>V24</th>
          <th>V25</th>
          <th>V26</th>
          <th>V27</th>
          <th>V28</th>
          <th>Amount</th>
          <th>Class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>284807.000</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>...</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>2.848e+05</td>
          <td>284807.000</td>
          <td>284807.000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>94813.860</td>
          <td>3.920e-15</td>
          <td>5.688e-16</td>
          <td>-8.769e-15</td>
          <td>2.782e-15</td>
          <td>-1.553e-15</td>
          <td>2.011e-15</td>
          <td>-1.694e-15</td>
          <td>-1.927e-16</td>
          <td>-3.137e-15</td>
          <td>...</td>
          <td>1.537e-16</td>
          <td>7.960e-16</td>
          <td>5.368e-16</td>
          <td>4.458e-15</td>
          <td>1.453e-15</td>
          <td>1.699e-15</td>
          <td>-3.660e-16</td>
          <td>-1.206e-16</td>
          <td>88.350</td>
          <td>0.002</td>
        </tr>
        <tr>
          <th>std</th>
          <td>47488.146</td>
          <td>1.959e+00</td>
          <td>1.651e+00</td>
          <td>1.516e+00</td>
          <td>1.416e+00</td>
          <td>1.380e+00</td>
          <td>1.332e+00</td>
          <td>1.237e+00</td>
          <td>1.194e+00</td>
          <td>1.099e+00</td>
          <td>...</td>
          <td>7.345e-01</td>
          <td>7.257e-01</td>
          <td>6.245e-01</td>
          <td>6.056e-01</td>
          <td>5.213e-01</td>
          <td>4.822e-01</td>
          <td>4.036e-01</td>
          <td>3.301e-01</td>
          <td>250.120</td>
          <td>0.042</td>
        </tr>
        <tr>
          <th>min</th>
          <td>0.000</td>
          <td>-5.641e+01</td>
          <td>-7.272e+01</td>
          <td>-4.833e+01</td>
          <td>-5.683e+00</td>
          <td>-1.137e+02</td>
          <td>-2.616e+01</td>
          <td>-4.356e+01</td>
          <td>-7.322e+01</td>
          <td>-1.343e+01</td>
          <td>...</td>
          <td>-3.483e+01</td>
          <td>-1.093e+01</td>
          <td>-4.481e+01</td>
          <td>-2.837e+00</td>
          <td>-1.030e+01</td>
          <td>-2.605e+00</td>
          <td>-2.257e+01</td>
          <td>-1.543e+01</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>54201.500</td>
          <td>-9.204e-01</td>
          <td>-5.985e-01</td>
          <td>-8.904e-01</td>
          <td>-8.486e-01</td>
          <td>-6.916e-01</td>
          <td>-7.683e-01</td>
          <td>-5.541e-01</td>
          <td>-2.086e-01</td>
          <td>-6.431e-01</td>
          <td>...</td>
          <td>-2.284e-01</td>
          <td>-5.424e-01</td>
          <td>-1.618e-01</td>
          <td>-3.546e-01</td>
          <td>-3.171e-01</td>
          <td>-3.270e-01</td>
          <td>-7.084e-02</td>
          <td>-5.296e-02</td>
          <td>5.600</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>84692.000</td>
          <td>1.811e-02</td>
          <td>6.549e-02</td>
          <td>1.798e-01</td>
          <td>-1.985e-02</td>
          <td>-5.434e-02</td>
          <td>-2.742e-01</td>
          <td>4.010e-02</td>
          <td>2.236e-02</td>
          <td>-5.143e-02</td>
          <td>...</td>
          <td>-2.945e-02</td>
          <td>6.782e-03</td>
          <td>-1.119e-02</td>
          <td>4.098e-02</td>
          <td>1.659e-02</td>
          <td>-5.214e-02</td>
          <td>1.342e-03</td>
          <td>1.124e-02</td>
          <td>22.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>139320.500</td>
          <td>1.316e+00</td>
          <td>8.037e-01</td>
          <td>1.027e+00</td>
          <td>7.433e-01</td>
          <td>6.119e-01</td>
          <td>3.986e-01</td>
          <td>5.704e-01</td>
          <td>3.273e-01</td>
          <td>5.971e-01</td>
          <td>...</td>
          <td>1.864e-01</td>
          <td>5.286e-01</td>
          <td>1.476e-01</td>
          <td>4.395e-01</td>
          <td>3.507e-01</td>
          <td>2.410e-01</td>
          <td>9.105e-02</td>
          <td>7.828e-02</td>
          <td>77.165</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>max</th>
          <td>172792.000</td>
          <td>2.455e+00</td>
          <td>2.206e+01</td>
          <td>9.383e+00</td>
          <td>1.688e+01</td>
          <td>3.480e+01</td>
          <td>7.330e+01</td>
          <td>1.206e+02</td>
          <td>2.001e+01</td>
          <td>1.559e+01</td>
          <td>...</td>
          <td>2.720e+01</td>
          <td>1.050e+01</td>
          <td>2.253e+01</td>
          <td>4.585e+00</td>
          <td>7.520e+00</td>
          <td>3.517e+00</td>
          <td>3.161e+01</td>
          <td>3.385e+01</td>
          <td>25691.160</td>
          <td>1.000</td>
        </tr>
      </tbody>
    </table>
    <p>8 rows × 31 columns</p>
    </div>



Let us check the number of fraud vs. non-fraud cases in the data set.

.. code:: ipython3

    class_names = {0:'Not Fraud', 1:'Fraud'}
    print(dataset.Class.value_counts().rename(index = class_names))


.. parsed-literal::

    Not Fraud    284315
    Fraud           492
    Name: Class, dtype: int64


The dataset is unbalanced with most of the transactions being non-fraud.

 ## 3.2. Data Visualization

.. code:: ipython3

    # histograms
    dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
    pyplot.show()



.. image:: output_23_0.png


Distribution of most of the variables are highly skewed. However, given
the variable names aren’t known, we don’t get much intuition from the
plot.

 ## 4. Data Preparation

.. code:: ipython3

    #Checking for any null values and removing the null values'''
    print('Null Values =',dataset.isnull().values.any())


.. parsed-literal::

    Null Values = False


There is no null in the data, and the data is already in the float
format, so there is no need to clean or categorise the data

 ## 4.2. Feature Selection

.. code:: ipython3

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    bestfeatures = SelectKBest( k=10)
    bestfeatures
    Y= dataset["Class"]
    X = dataset.loc[:, dataset.columns != 'Class']
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features


.. parsed-literal::

       Specs      Score
    17   V17  33979.169
    14   V14  28695.548
    12   V12  20749.822
    10   V10  14057.980
    16   V16  11443.349
    3     V3  11014.508
    7     V7  10349.605
    11   V11   6999.355
    4     V4   5163.832
    18   V18   3584.381


Although some of the features are relevant, feature selection is not
given significant preference

 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split and Evaluation Metrics

.. code:: ipython3

    # split out validation dataset for the end
    Y= dataset["Class"]
    X = dataset.loc[:, dataset.columns != 'Class']
    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    scoring = 'accuracy'

 ## 5.2. Checking Models and Algorithms

.. code:: ipython3

    # test options for classification
    num_folds = 10
    seed = 7

.. code:: ipython3

    # spot check some basic Classification algorithms
    #Given Data is huge, some of the slower classification algorithms are commented
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # #Neural Network
    # models.append(('NN', MLPClassifier()))
    # # #Ensable Models
    # # Boosting methods
    # models.append(('AB', AdaBoostClassifier()))
    # models.append(('GBM', GradientBoostingClassifier()))
    # # Bagging methods
    # models.append(('RF', RandomForestClassifier()))
    # models.append(('ET', ExtraTreesClassifier()))

.. code:: ipython3

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


.. parsed-literal::

    LR: 0.998942 (0.000229)
    LDA: 0.999364 (0.000136)
    KNN: 0.998310 (0.000290)
    CART: 0.999175 (0.000193)


.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    fig.set_size_inches(8,4)
    pyplot.show()



.. image:: output_38_0.png


The accuracy is very high, given that accuracy focusses on the overall
no fraud case, but lets check how well it predicts the fraud case.
Choosing one of the model CART from the results above

.. code:: ipython3

    # prepare model
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)




.. parsed-literal::

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')



.. code:: ipython3

    # estimate accuracy on validation set
    #rescaledValidationX = scaler.transform(X_validation)
    rescaledValidationX = X_validation
    predictions = model.predict(rescaledValidationX)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


.. parsed-literal::

    0.9992275552122467
    [[56839    23]
     [   21    79]]
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     56862
               1       0.77      0.79      0.78       100

        accuracy                           1.00     56962
       macro avg       0.89      0.89      0.89     56962
    weighted avg       1.00      1.00      1.00     56962



.. code:: ipython3

    df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation), index = np.unique(Y_validation))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x20b99300898>




.. image:: output_42_1.png


Although results are good, but still 21 out of 100 frauds aren’t caught.
So, we should focus on *recall*, which is a metric which minimises false
negative.

 ## 6. Model Tuning

 ## 6.1. Model Tuning by choosing correct evaluation metric Evaluation
Metric recall is selected, which is a metric which minimises false
negative.

.. code:: ipython3

    scoring = 'recall'

.. code:: ipython3

    # spot check some basic Classification algorithms
    #Given Data is huge, some of the slower classification algorithms are commented
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # #Neural Network
    # models.append(('NN', MLPClassifier()))
    # # #Ensable Models
    # # Boosting methods
    # models.append(('AB', AdaBoostClassifier()))
    # models.append(('GBM', GradientBoostingClassifier()))
    # # Bagging methods
    # models.append(('RF', RandomForestClassifier()))
    # models.append(('ET', ExtraTreesClassifier()))

.. code:: ipython3

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


.. parsed-literal::

    LR: 0.595470 (0.089743)
    LDA: 0.758283 (0.045450)
    KNN: 0.023882 (0.019671)
    CART: 0.735192 (0.073650)


Given the LDA has the best recall out of all the models, it is used to
ealuate the test set

.. code:: ipython3

    # prepare model
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)




.. parsed-literal::

    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                               solver='svd', store_covariance=False, tol=0.0001)



.. code:: ipython3

    # estimate accuracy on validation set
    #rescaledValidationX = scaler.transform(X_validation)
    rescaledValidationX = X_validation
    predictions = model.predict(rescaledValidationX)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


.. parsed-literal::

    0.9995435553526912
    [[56854     8]
     [   18    82]]
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     56862
               1       0.91      0.82      0.86       100

        accuracy                           1.00     56962
       macro avg       0.96      0.91      0.93     56962
    weighted avg       1.00      1.00      1.00     56962



.. code:: ipython3

    df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation), index = np.unique(Y_validation))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x20b99399978>




.. image:: output_52_1.png


LDA performs much better with only 18 out of cases of fraud not caught.
Additionally, false positives are less as well. However, there are still
18 fraud cases in the test set which aren’t caught. This will be further
taken care in the following section.

 ## 6.2. Model Tuning for balancing the sample by Random Under Sampling
In this phase of the project we will implement “Random Under Sampling”
which basically consists of removing data in order to have a more
balanced dataset and thus avoiding our models to overfitting.

Steps: 1. The first thing we have to do is determine how imbalanced is
our class (use “value_counts()” on the class column to determine the
amount for each label) 2. Once we determine how many instances are
considered fraud transactions (Fraud = “1”) , we should bring the
non-fraud transactions to the same amount as fraud transactions
(assuming we want a 50/50 ratio), this will be equivalent to 492 cases
of fraud and 492 cases of non-fraud transactions. 3. After implementing
this technique, we have a sub-sample of our dataframe with a 50/50 ratio
with regards to our classes. Then the next step we will implement is to
shuffle the data to see if our models can maintain a certain accuracy
everytime we run this script.

Note: The main issue with “Random Under-Sampling” is that we run the
risk that our classification models will not perform as accurate as we
would like to since there is a great deal of information loss (bringing
492 non-fraud transaction from 284,315 non-fraud transaction)

.. code:: ipython3

    Y_train.head()




.. parsed-literal::

    44828     0
    221877    0
    278826    0
    149792    0
    226041    0
    Name: Class, dtype: int64



.. code:: ipython3

    df = pd.concat([X_train, Y_train], axis=1)
    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    df_new = normal_distributed_df.sample(frac=1, random_state=42)
    # split out validation dataset for the end
    Y_train_new= df_new["Class"]
    X_train_new = df_new.loc[:, dataset.columns != 'Class']

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
          <th>Time</th>
          <th>V1</th>
          <th>V2</th>
          <th>V3</th>
          <th>V4</th>
          <th>V5</th>
          <th>V6</th>
          <th>V7</th>
          <th>V8</th>
          <th>V9</th>
          <th>...</th>
          <th>V21</th>
          <th>V22</th>
          <th>V23</th>
          <th>V24</th>
          <th>V25</th>
          <th>V26</th>
          <th>V27</th>
          <th>V28</th>
          <th>Amount</th>
          <th>Class</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.0</td>
          <td>-1.360</td>
          <td>-0.073</td>
          <td>2.536</td>
          <td>1.378</td>
          <td>-0.338</td>
          <td>0.462</td>
          <td>0.240</td>
          <td>0.099</td>
          <td>0.364</td>
          <td>...</td>
          <td>-0.018</td>
          <td>0.278</td>
          <td>-0.110</td>
          <td>0.067</td>
          <td>0.129</td>
          <td>-0.189</td>
          <td>0.134</td>
          <td>-0.021</td>
          <td>149.62</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.0</td>
          <td>1.192</td>
          <td>0.266</td>
          <td>0.166</td>
          <td>0.448</td>
          <td>0.060</td>
          <td>-0.082</td>
          <td>-0.079</td>
          <td>0.085</td>
          <td>-0.255</td>
          <td>...</td>
          <td>-0.226</td>
          <td>-0.639</td>
          <td>0.101</td>
          <td>-0.340</td>
          <td>0.167</td>
          <td>0.126</td>
          <td>-0.009</td>
          <td>0.015</td>
          <td>2.69</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.0</td>
          <td>-1.358</td>
          <td>-1.340</td>
          <td>1.773</td>
          <td>0.380</td>
          <td>-0.503</td>
          <td>1.800</td>
          <td>0.791</td>
          <td>0.248</td>
          <td>-1.515</td>
          <td>...</td>
          <td>0.248</td>
          <td>0.772</td>
          <td>0.909</td>
          <td>-0.689</td>
          <td>-0.328</td>
          <td>-0.139</td>
          <td>-0.055</td>
          <td>-0.060</td>
          <td>378.66</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.0</td>
          <td>-0.966</td>
          <td>-0.185</td>
          <td>1.793</td>
          <td>-0.863</td>
          <td>-0.010</td>
          <td>1.247</td>
          <td>0.238</td>
          <td>0.377</td>
          <td>-1.387</td>
          <td>...</td>
          <td>-0.108</td>
          <td>0.005</td>
          <td>-0.190</td>
          <td>-1.176</td>
          <td>0.647</td>
          <td>-0.222</td>
          <td>0.063</td>
          <td>0.061</td>
          <td>123.50</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.0</td>
          <td>-1.158</td>
          <td>0.878</td>
          <td>1.549</td>
          <td>0.403</td>
          <td>-0.407</td>
          <td>0.096</td>
          <td>0.593</td>
          <td>-0.271</td>
          <td>0.818</td>
          <td>...</td>
          <td>-0.009</td>
          <td>0.798</td>
          <td>-0.137</td>
          <td>0.141</td>
          <td>-0.206</td>
          <td>0.502</td>
          <td>0.219</td>
          <td>0.215</td>
          <td>69.99</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 31 columns</p>
    </div>



.. code:: ipython3

    print('Distribution of the Classes in the subsample dataset')
    print(df_new['Class'].value_counts()/len(df_new))
    sns.countplot('Class', data=df_new)
    pyplot.title('Equally Distributed Classes', fontsize=14)
    pyplot.show()


.. parsed-literal::

    Distribution of the Classes in the subsample dataset
    1    0.5
    0    0.5
    Name: Class, dtype: float64



.. image:: output_57_1.png


Now that we have our dataframe correctly balanced, we can go further
with our analysis and data preprocessing. Given the total number of data
points are around 900, we try all the Models including Deep Learning
Models. However, given the data is balanced, the metric used here is
accuracy, as it focuses on both false positive and false negative.

.. code:: ipython3

    scoring='accuracy'

.. code:: ipython3

    # spot check the algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    #Neural Network
    models.append(('NN', MLPClassifier()))
    # #Ensable Models
    # Boosting methods
    models.append(('AB', AdaBoostClassifier()))
    models.append(('GBM', GradientBoostingClassifier()))
    # Bagging methods
    models.append(('RF', RandomForestClassifier()))
    models.append(('ET', ExtraTreesClassifier()))

.. code:: ipython3

    #Writing the Deep Learning Classifier in case the Deep Learning Flag is Set to True
    #Set the following Flag to 1 if the Deep LEarning Models Flag has to be enabled
    EnableDLModelsFlag = 1
    if EnableDLModelsFlag == 1 :
        # Function to create model, required for KerasClassifier
        def create_model(neurons=12, activation='relu', learn_rate = 0.01, momentum=0):
            # create model
            model = Sequential()
            model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation=activation))
            model.add(Dense(32, activation=activation))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            optimizer = SGD(lr=learn_rate, momentum=momentum)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        models.append(('DNN', KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)))

.. code:: ipython3

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train_new, Y_train_new, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


.. parsed-literal::

    LR: 0.931911 (0.024992)
    LDA: 0.905473 (0.027422)
    KNN: 0.648258 (0.044550)
    CART: 0.907565 (0.022669)
    NB: 0.860771 (0.027234)
    SVM: 0.522356 (0.048395)
    NN: 0.648712 (0.100137)
    AB: 0.924830 (0.024068)
    GBM: 0.934982 (0.015132)
    RF: 0.932931 (0.015859)
    ET: 0.931962 (0.031043)
    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    DNN: 0.498011 (0.050742)


.. code:: ipython3

    # compare algorithms
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    fig.set_size_inches(8,4)
    pyplot.show()



.. image:: output_63_0.png


Given that GBM is the best model out of all the models, a grid search is
performed for GBM model by varing number of estimators and maximum
depth.

.. code:: ipython3

    # Grid Search: GradientBoosting Tuning
    '''
    n_estimators : int (default=100)
        The number of boosting stages to perform.
        Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators.
        The maximum depth limits the number of nodes in the tree.
        Tune this parameter for best performance; the best value depends on the interaction of the input variables.

    '''
    n_estimators = [20,180,1000]
    max_depth= [2, 3,5]
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    model = GradientBoostingClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train_new, Y_train_new)

    #Print Results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    ranks = grid_result.cv_results_['rank_test_score']
    for mean, stdev, param, rank in zip(means, stds, params, ranks):
        print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))


.. parsed-literal::

    Best: 0.936992 using {'max_depth': 5, 'n_estimators': 1000}
    #3 0.931911 (0.016958) with: {'max_depth': 2, 'n_estimators': 20}
    #6 0.929878 (0.017637) with: {'max_depth': 2, 'n_estimators': 180}
    #9 0.924797 (0.021358) with: {'max_depth': 2, 'n_estimators': 1000}
    #6 0.929878 (0.020476) with: {'max_depth': 3, 'n_estimators': 20}
    #3 0.931911 (0.011120) with: {'max_depth': 3, 'n_estimators': 180}
    #3 0.931911 (0.017026) with: {'max_depth': 3, 'n_estimators': 1000}
    #8 0.928862 (0.022586) with: {'max_depth': 5, 'n_estimators': 20}
    #2 0.934959 (0.015209) with: {'max_depth': 5, 'n_estimators': 180}
    #1 0.936992 (0.012639) with: {'max_depth': 5, 'n_estimators': 1000}


.. code:: ipython3

    # prepare model
    model = GradientBoostingClassifier(max_depth= 5, n_estimators = 1000)
    model.fit(X_train_new, Y_train_new)




.. parsed-literal::

    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=5,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=1000,
                               n_iter_no_change=None, presort='auto',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)



.. code:: ipython3

    # estimate accuracy on Original validation set
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


.. parsed-literal::

    0.9668199852533268
    [[54972  1890]
     [    0   100]]
                  precision    recall  f1-score   support

               0       1.00      0.97      0.98     56862
               1       0.05      1.00      0.10       100

        accuracy                           0.97     56962
       macro avg       0.53      0.98      0.54     56962
    weighted avg       1.00      0.97      0.98     56962



.. code:: ipython3

    df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation), index = np.unique(Y_validation))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x26e0cc0bb70>




.. image:: output_68_1.png


The results on the test set are really good and the model performs much
better with no case of fraud not caught.

**Conclusion**:

Choosing the right metric lead to an enhancement in the fraud cases
detected correctly. Under-sampling lead to a significant improvement as
all the fraud cases in the test set are correctly identified post
under-sampling.

Under-sampling came with a tradeoff though. In the under-sampled data
our model is unable to detect for a large number of cases non-fraud
transactions correctly and instead, misclassifies those non-fraud
transactions as fraud cases.
