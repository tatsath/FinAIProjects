.. _Clss_lend:



Loan Default Probability
========================

The goal of this case study is to build a machine learning model to
predict the probability that a loan will default.

Content
-------

-  `1. Problem Definition <#0>`__
-  `2. Getting Started - Load Libraries and Dataset <#1>`__

   -  `2.1. Load Libraries <#1.1>`__
   -  `2.2. Load Dataset <#1.2>`__

-  `3. Data Preparation and Feature Selection <#2>`__

   -  `3.1. Preparing the predicted variable <#2.1>`__
   -  `3.2. Feature Selection-Limit the Feature Space <#2.2>`__

      -  `3.2.1. Features elimination by significant missing
         values <#2.2.1>`__
      -  `3.2.2. Features elimination based on the
         intutiveness <#2.2.2>`__
      -  `3.2.3. Features elimination based on the
         correlation <#2.2.3>`__

-  `4. Feature Engineering and Exploratory Analysis <#3>`__

   -  `4.1 Feature Analysis and Exploration <#3.1>`__

      -  `4.1.1. Analysing the categorical features <#3.1.1>`__
      -  `4.1.2 Analysing the continuous features <#3.1.2>`__

   -  `4.2.Encoding Categorical Data <#3.2>`__
   -  `4.3.Sampling Data <#3.3>`__

-  `5.Evaluate Algorithms and Models <#4>`__

   -  `5.1. Train/Test Split <#4.1>`__
   -  `5.2. Test Options and Evaluation Metrics <#4.2>`__
   -  `5.3. Compare Models and Algorithms <#4.3>`__

-  `6. Model Tuning and Grid Search <#5>`__
-  `7. Finalize the Model <#6>`__

   -  `7.1. Results on test dataset <#6.1>`__
   -  `7.1. Variable Intuition/Feature Selection <#6.2>`__
   -  `7.3. Save model for later use <#6.3>`__

 # 1. Problem Definition

The problem is defined in the classification framework, where the
predicted variable is “Charge-Off”. A charge-off is a debt that a
creditor has given up trying to collect on after you’ve missed payments
for several months. The predicted variable takes value 1 in case of
charge-off and 0 otherwise.

This case study aims to analyze data for loans through 2007-2017Q3 from
Lending Club available on Kaggle. Dataset contains over 887 thousand
observations and 150 variables among which one is describing the loan
status.

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

Note : Due to limit in the github for the data size, a sample of the data has been loaded in the jupyter notebook repository of this book. However, all the subsequent results in this jupyter notebook is with actual data (~1GB) under https://www.kaggle.com/mlfinancebook/lending-club-loans-data. You should load the full data in case you want to reproduce the results.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # load dataset
    loans = pd.read_csv('LoansData_sample.csv.gz', compression='gzip', encoding='utf-8')
    #loans = pd.read_csv('LoansData.csv.gz', compression='gzip', low_memory=True) #Use this for the actual data

.. code:: ipython3

    dataset = loans

.. code:: ipython3

    #Diable the warnings
    import warnings
    warnings.filterwarnings('ignore')

.. code:: ipython3

    type(dataset)




.. parsed-literal::

    pandas.core.frame.DataFrame



 ## 3. Data Preparation and Feature Selection

 ## 3.1. Preparing the predicted variable

.. code:: ipython3

    # We're going to try to predict the loan_status variable. What are the value counts for this variable
    dataset['loan_status'].value_counts(dropna=False)




.. parsed-literal::

    Current                                                788950
    Fully Paid                                             646902
    Charged Off                                            168084
    Late (31-120 days)                                      23763
    In Grace Period                                         10474
    Late (16-30 days)                                        5786
    Does not meet the credit policy. Status:Fully Paid       1988
    Does not meet the credit policy. Status:Charged Off       761
    Default                                                    70
    NaN                                                        23
    Name: loan_status, dtype: int64



We’re going to try to learn differences in the features between
completed loans that have been fully paid or charged off. We won’t
consider loans that are current, don’t meet the credit policy,
defaulted, or have a missing status. So we only keep the loans with
status “Fully Paid” or “Charged Off.”

.. code:: ipython3

    dataset = dataset.loc[dataset['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    dataset['loan_status'].value_counts(dropna=False)

    dataset['loan_status'].value_counts(normalize=True, dropna=False)




.. parsed-literal::

    Fully Paid     0.793758
    Charged Off    0.206242
    Name: loan_status, dtype: float64



About 79% of the remaining loans have been fully paid and 21% have
charged off, so we have a somewhat unbalanced classification problem.

.. code:: ipython3

    dataset['charged_off'] = (dataset['loan_status'] == 'Charged Off').apply(np.uint8)
    dataset.drop('loan_status', axis=1, inplace=True)

 ## 3.2. Feature Selection-Limit the Feature Space

The full dataset has 150 features for each loan. We’ll eliminate
features in following steps using three different approaches: \*
Eliminate feature that have more than 30% missing values. \* Eliminate
features that are unintuitive based on subjective judgement. \*
Eliminate features with low correlation with the predicted variable

 ### 3.2.1. Features elimination by significant missing values

First calculating the percentage of missing data for each feature:

.. code:: ipython3

    missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

    missing_fractions.head(10)




.. parsed-literal::

    next_pymnt_d                                  1.000000
    member_id                                     1.000000
    orig_projected_additional_accrued_interest    0.999876
    sec_app_mths_since_last_major_derog           0.999628
    hardship_dpd                                  0.999275
    hardship_reason                               0.999275
    hardship_status                               0.999275
    deferral_term                                 0.999275
    hardship_amount                               0.999275
    hardship_start_date                           0.999275
    dtype: float64



.. code:: ipython3

    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    print(drop_list)


.. parsed-literal::

    ['all_util', 'annual_inc_joint', 'debt_settlement_flag_date', 'deferral_term', 'desc', 'dti_joint', 'hardship_amount', 'hardship_dpd', 'hardship_end_date', 'hardship_last_payment_amount', 'hardship_length', 'hardship_loan_status', 'hardship_payoff_balance_amount', 'hardship_reason', 'hardship_start_date', 'hardship_status', 'hardship_type', 'il_util', 'inq_fi', 'inq_last_12m', 'max_bal_bc', 'member_id', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 'mths_since_rcnt_il', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 'next_pymnt_d', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m', 'orig_projected_additional_accrued_interest', 'payment_plan_start_date', 'revol_bal_joint', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'sec_app_earliest_cr_line', 'sec_app_fico_range_high', 'sec_app_fico_range_low', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_mths_since_last_major_derog', 'sec_app_num_rev_accts', 'sec_app_open_acc', 'sec_app_open_act_il', 'sec_app_revol_util', 'settlement_amount', 'settlement_date', 'settlement_percentage', 'settlement_status', 'settlement_term', 'total_bal_il', 'total_cu_tl', 'verification_status_joint']


.. code:: ipython3

    len(drop_list)




.. parsed-literal::

    58



.. code:: ipython3

    dataset.drop(labels=drop_list, axis=1, inplace=True)
    dataset.shape




.. parsed-literal::

    (814986, 92)



 ### 3.2.2. Features elimination based on the intutiveness

In order to filter the features further we check the description in the
data dictionary and keep the features that are intuitive on the basis of
subjective judgement.

We examine the LendingClub website and Data Dictionary to determine
which features would have been available to potential investors. Here’s
the list of features we currently have, in alphabetical order:

.. code:: ipython3

    print(sorted(dataset.columns))


.. parsed-literal::

    ['acc_now_delinq', 'acc_open_past_24mths', 'addr_state', 'annual_inc', 'application_type', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'charged_off', 'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 'delinq_2yrs', 'delinq_amnt', 'disbursement_method', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'funded_amnt', 'funded_amnt_inv', 'grade', 'hardship_flag', 'home_ownership', 'id', 'initial_list_status', 'inq_last_6mths', 'installment', 'int_rate', 'issue_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d', 'loan_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'open_acc', 'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'policy_code', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'pymnt_plan', 'recoveries', 'revol_bal', 'revol_util', 'sub_grade', 'tax_liens', 'term', 'title', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_acc', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim', 'verification_status', 'zip_code']


The list that is kept is as follows:

.. code:: ipython3

    keep_list = ['charged_off','funded_amnt','addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status', 'installment', 'int_rate', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code','last_pymnt_amnt','num_actv_rev_tl', 'mo_sin_rcnt_rev_tl_op','mo_sin_old_rev_tl_op',"bc_util","bc_open_to_buy","avg_cur_bal","acc_open_past_24mths" ]

    len(keep_list)




.. parsed-literal::

    40



.. code:: ipython3

    drop_list = [col for col in dataset.columns if col not in keep_list]

    dataset.drop(labels=drop_list, axis=1, inplace=True)

    dataset.shape




.. parsed-literal::

    (814986, 39)



 ### 3.2.3. Features elimination based on the correlation

.. code:: ipython3

    correlation = dataset.corr()
    correlation_chargeOff = abs(correlation['charged_off'])

.. code:: ipython3

    correlation_chargeOff.sort_values(ascending=False)




.. parsed-literal::

    charged_off              1.000000
    last_pymnt_amnt          0.381359
    int_rate                 0.247815
    fico_range_low           0.139430
    fico_range_high          0.139428
    dti                      0.123031
    acc_open_past_24mths     0.098985
    bc_open_to_buy           0.086896
    avg_cur_bal              0.085777
    num_actv_rev_tl          0.077211
    bc_util                  0.077132
    mort_acc                 0.077086
    revol_util               0.072185
    funded_amnt              0.064258
    loan_amnt                0.064139
    mo_sin_rcnt_rev_tl_op    0.053469
    mo_sin_old_rev_tl_op     0.048529
    annual_inc               0.046685
    installment              0.046291
    open_acc                 0.034652
    pub_rec                  0.023105
    pub_rec_bankruptcies     0.017314
    revol_bal                0.013160
    total_acc                0.011187
    Name: charged_off, dtype: float64



.. code:: ipython3

    drop_list_corr = sorted(list(correlation_chargeOff[correlation_chargeOff < 0.03].index))
    print(drop_list_corr)


.. parsed-literal::

    ['pub_rec', 'pub_rec_bankruptcies', 'revol_bal', 'total_acc']


.. code:: ipython3

    dataset.drop(labels=drop_list_corr, axis=1, inplace=True)
    dataset.shape




.. parsed-literal::

    (814986, 35)



 # 4. Feature Engineering and Exploratory Analysis

.. code:: ipython3

    #Descriptive Statistics
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
          <th>loan_amnt</th>
          <th>funded_amnt</th>
          <th>int_rate</th>
          <th>installment</th>
          <th>annual_inc</th>
          <th>dti</th>
          <th>fico_range_low</th>
          <th>fico_range_high</th>
          <th>open_acc</th>
          <th>revol_util</th>
          <th>last_pymnt_amnt</th>
          <th>acc_open_past_24mths</th>
          <th>avg_cur_bal</th>
          <th>bc_open_to_buy</th>
          <th>bc_util</th>
          <th>mo_sin_old_rev_tl_op</th>
          <th>mo_sin_rcnt_rev_tl_op</th>
          <th>mort_acc</th>
          <th>num_actv_rev_tl</th>
          <th>charged_off</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>814986.000000</td>
          <td>814986.000000</td>
          <td>814986.000000</td>
          <td>814986.000000</td>
          <td>8.149860e+05</td>
          <td>814950.000000</td>
          <td>814986.000000</td>
          <td>814986.000000</td>
          <td>814986.000000</td>
          <td>814496.000000</td>
          <td>814986.000000</td>
          <td>767705.000000</td>
          <td>747447.000000</td>
          <td>759810.00000</td>
          <td>759321.000000</td>
          <td>747458.000000</td>
          <td>747458.000000</td>
          <td>767705.000000</td>
          <td>747459.000000</td>
          <td>814986.000000</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>14315.458210</td>
          <td>14301.566929</td>
          <td>13.490993</td>
          <td>436.749624</td>
          <td>7.523039e+04</td>
          <td>17.867719</td>
          <td>695.603151</td>
          <td>699.603264</td>
          <td>11.521099</td>
          <td>53.031137</td>
          <td>5918.144144</td>
          <td>4.716176</td>
          <td>13519.786576</td>
          <td>9464.94483</td>
          <td>61.575664</td>
          <td>180.843182</td>
          <td>12.705577</td>
          <td>1.758707</td>
          <td>5.658872</td>
          <td>0.206242</td>
        </tr>
        <tr>
          <th>std</th>
          <td>8499.799241</td>
          <td>8492.964986</td>
          <td>4.618486</td>
          <td>255.732093</td>
          <td>6.524373e+04</td>
          <td>8.856477</td>
          <td>31.352251</td>
          <td>31.352791</td>
          <td>5.325064</td>
          <td>24.320981</td>
          <td>7279.949481</td>
          <td>3.152369</td>
          <td>16221.882463</td>
          <td>14575.87033</td>
          <td>27.871170</td>
          <td>92.192939</td>
          <td>15.654277</td>
          <td>2.081730</td>
          <td>3.215863</td>
          <td>0.404606</td>
        </tr>
        <tr>
          <th>min</th>
          <td>500.000000</td>
          <td>500.000000</td>
          <td>5.320000</td>
          <td>4.930000</td>
          <td>0.000000e+00</td>
          <td>-1.000000</td>
          <td>625.000000</td>
          <td>629.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.00000</td>
          <td>0.000000</td>
          <td>2.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>8000.000000</td>
          <td>8000.000000</td>
          <td>9.990000</td>
          <td>251.400000</td>
          <td>4.500000e+04</td>
          <td>11.640000</td>
          <td>670.000000</td>
          <td>674.000000</td>
          <td>8.000000</td>
          <td>35.000000</td>
          <td>446.922500</td>
          <td>2.000000</td>
          <td>3119.000000</td>
          <td>1312.00000</td>
          <td>40.800000</td>
          <td>117.000000</td>
          <td>4.000000</td>
          <td>0.000000</td>
          <td>3.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>12000.000000</td>
          <td>12000.000000</td>
          <td>13.110000</td>
          <td>377.040000</td>
          <td>6.500000e+04</td>
          <td>17.360000</td>
          <td>690.000000</td>
          <td>694.000000</td>
          <td>11.000000</td>
          <td>53.700000</td>
          <td>2864.715000</td>
          <td>4.000000</td>
          <td>7508.000000</td>
          <td>4261.00000</td>
          <td>65.400000</td>
          <td>164.000000</td>
          <td>8.000000</td>
          <td>1.000000</td>
          <td>5.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>20000.000000</td>
          <td>20000.000000</td>
          <td>16.290000</td>
          <td>576.290000</td>
          <td>9.000000e+04</td>
          <td>23.630000</td>
          <td>710.000000</td>
          <td>714.000000</td>
          <td>14.000000</td>
          <td>71.900000</td>
          <td>9193.050000</td>
          <td>6.000000</td>
          <td>18827.000000</td>
          <td>11343.00000</td>
          <td>86.000000</td>
          <td>228.000000</td>
          <td>15.000000</td>
          <td>3.000000</td>
          <td>7.000000</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>max</th>
          <td>40000.000000</td>
          <td>40000.000000</td>
          <td>30.990000</td>
          <td>1714.540000</td>
          <td>9.550000e+06</td>
          <td>999.000000</td>
          <td>845.000000</td>
          <td>850.000000</td>
          <td>90.000000</td>
          <td>892.300000</td>
          <td>42148.530000</td>
          <td>56.000000</td>
          <td>958084.000000</td>
          <td>559912.00000</td>
          <td>339.600000</td>
          <td>842.000000</td>
          <td>372.000000</td>
          <td>51.000000</td>
          <td>57.000000</td>
          <td>1.000000</td>
        </tr>
      </tbody>
    </table>
    </div>



 ## 4.1 Feature Analysis and Exploration

 ### 4.1.1. Analysing the categorical features

.. code:: ipython3

    dataset[['id','emp_title','title','zip_code']].describe()




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
          <th>id</th>
          <th>emp_title</th>
          <th>title</th>
          <th>zip_code</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>814986</td>
          <td>766415</td>
          <td>807068</td>
          <td>814986</td>
        </tr>
        <tr>
          <th>unique</th>
          <td>814986</td>
          <td>280473</td>
          <td>60298</td>
          <td>925</td>
        </tr>
        <tr>
          <th>top</th>
          <td>14680062</td>
          <td>Teacher</td>
          <td>Debt consolidation</td>
          <td>945xx</td>
        </tr>
        <tr>
          <th>freq</th>
          <td>1</td>
          <td>11351</td>
          <td>371874</td>
          <td>9517</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #Ids are all unique and there are too many job titles and titles and zipcode,
    #these column is dropped The ID is not useful for modeling.
    dataset.drop(['id','emp_title','title','zip_code'], axis=1, inplace=True)

Feature- Term
^^^^^^^^^^^^^

.. code:: ipython3

    #Data Dictionary: "The number of payments on the loan. Values are in months and can be either 36 or 60.".
    #The 60 Months loans are more likelely to charge off
    #Convert term to integers
    dataset['term'] = dataset['term'].apply(lambda s: np.int8(s.split()[0]))

.. code:: ipython3

    dataset.groupby('term')['charged_off'].value_counts(normalize=True).loc[:,1]




.. parsed-literal::

    term
    36    0.165710
    60    0.333793
    Name: charged_off, dtype: float64



Loans with five-year periods are more than twice as likely to charge-off
as loans with three-year periods.

Feature- Employement Length
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    dataset['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)

    dataset['emp_length'].replace('< 1 year', '0 years', inplace=True)

    def emp_length_to_int(s):
        if pd.isnull(s):
            return s
        else:
            return np.int8(s.split()[0])

    dataset['emp_length'] = dataset['emp_length'].apply(emp_length_to_int)


.. code:: ipython3

    charge_off_rates = dataset.groupby('emp_length')['charged_off'].value_counts(normalize=True).loc[:,1]
    sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#5975A4', saturation=1)




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x25690281470>




.. image:: output_52_1.png


Loan status does not appear to vary much with employment length on
average, hence it is dropped

.. code:: ipython3

    dataset.drop(['emp_length'], axis=1, inplace=True)

Feature : Subgrade
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    charge_off_rates = dataset.groupby('sub_grade')['charged_off'].value_counts(normalize=True).loc[:,1]
    sns.set(rc={'figure.figsize':(12,5)})
    sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#5975A4', saturation=1)





.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x2569e8cc668>




.. image:: output_56_1.png


There’s a clear trend of higher probability of charge-off as the
subgrade worsens.

.. code:: ipython3

    dataset['earliest_cr_line'] = dataset['earliest_cr_line'].apply(lambda s: int(s[-4:]))

 ### 4.1.2. Analysing the continuous features

Feature : Annual Income
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    dataset[['annual_inc']].describe()




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
          <th>annual_inc</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>count</th>
          <td>8.149860e+05</td>
        </tr>
        <tr>
          <th>mean</th>
          <td>7.523039e+04</td>
        </tr>
        <tr>
          <th>std</th>
          <td>6.524373e+04</td>
        </tr>
        <tr>
          <th>min</th>
          <td>0.000000e+00</td>
        </tr>
        <tr>
          <th>25%</th>
          <td>4.500000e+04</td>
        </tr>
        <tr>
          <th>50%</th>
          <td>6.500000e+04</td>
        </tr>
        <tr>
          <th>75%</th>
          <td>9.000000e+04</td>
        </tr>
        <tr>
          <th>max</th>
          <td>9.550000e+06</td>
        </tr>
      </tbody>
    </table>
    </div>



Annual income ranges from 0 to 9,550,000, with a median of $65,000.
Because of the large range of incomes, we should take a log transform of
the annual income variable.

.. code:: ipython3

    dataset['log_annual_inc'] = dataset['annual_inc'].apply(lambda x: np.log10(x+1))
    dataset.drop('annual_inc', axis=1, inplace=True)

FICO Scores
^^^^^^^^^^^

.. code:: ipython3

    dataset[['fico_range_low','fico_range_high']].corr()




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
          <th>fico_range_low</th>
          <th>fico_range_high</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>fico_range_low</th>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>fico_range_high</th>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Given that the correlation between fico low and high is 1 it is
preferred to keep only one feature which is average of FICO Scores

.. code:: ipython3

    dataset['fico_score'] = 0.5*dataset['fico_range_low'] + 0.5*dataset['fico_range_high']

    dataset.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)

.. code:: ipython3

    dataset['charged_off'].value_counts()




.. parsed-literal::

    0    646902
    1    168084
    Name: charged_off, dtype: int64



 ## 4.2. Encoding Categorical Data

.. code:: ipython3

    from sklearn.preprocessing import LabelEncoder

.. code:: ipython3

    # Categorical boolean mask
    categorical_feature_mask = dataset.dtypes==object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = dataset.columns[categorical_feature_mask].tolist()

.. code:: ipython3

    categorical_cols




.. parsed-literal::

    ['grade',
     'sub_grade',
     'home_ownership',
     'verification_status',
     'purpose',
     'addr_state',
     'initial_list_status',
     'application_type']



.. code:: ipython3

    le = LabelEncoder()
    # apply le on categorical feature columns
    dataset[categorical_cols] = dataset[categorical_cols].apply(lambda col: le.fit_transform(col))
    dataset[categorical_cols].head(10)




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
          <th>grade</th>
          <th>sub_grade</th>
          <th>home_ownership</th>
          <th>verification_status</th>
          <th>purpose</th>
          <th>addr_state</th>
          <th>initial_list_status</th>
          <th>application_type</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2</td>
          <td>10</td>
          <td>5</td>
          <td>1</td>
          <td>2</td>
          <td>45</td>
          <td>1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>2</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>4</td>
          <td>1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>15</td>
          <td>5</td>
          <td>1</td>
          <td>1</td>
          <td>24</td>
          <td>1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>12</td>
          <td>5</td>
          <td>1</td>
          <td>2</td>
          <td>3</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2</td>
          <td>12</td>
          <td>5</td>
          <td>1</td>
          <td>2</td>
          <td>31</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>1</td>
          <td>9</td>
          <td>1</td>
          <td>1</td>
          <td>4</td>
          <td>23</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>1</td>
          <td>8</td>
          <td>4</td>
          <td>2</td>
          <td>2</td>
          <td>45</td>
          <td>1</td>
          <td>0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>2</td>
          <td>13</td>
          <td>5</td>
          <td>1</td>
          <td>1</td>
          <td>47</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>1</td>
          <td>8</td>
          <td>5</td>
          <td>0</td>
          <td>2</td>
          <td>20</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>10</th>
          <td>1</td>
          <td>9</td>
          <td>5</td>
          <td>2</td>
          <td>2</td>
          <td>22</td>
          <td>0</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

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
          <th>loan_amnt</th>
          <th>funded_amnt</th>
          <th>term</th>
          <th>int_rate</th>
          <th>installment</th>
          <th>grade</th>
          <th>sub_grade</th>
          <th>home_ownership</th>
          <th>verification_status</th>
          <th>purpose</th>
          <th>...</th>
          <th>avg_cur_bal</th>
          <th>bc_open_to_buy</th>
          <th>bc_util</th>
          <th>mo_sin_old_rev_tl_op</th>
          <th>mo_sin_rcnt_rev_tl_op</th>
          <th>mort_acc</th>
          <th>num_actv_rev_tl</th>
          <th>charged_off</th>
          <th>log_annual_inc</th>
          <th>fico_score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>15000.0</td>
          <td>15000.0</td>
          <td>60</td>
          <td>12.39</td>
          <td>336.64</td>
          <td>2</td>
          <td>10</td>
          <td>5</td>
          <td>1</td>
          <td>2</td>
          <td>...</td>
          <td>29828.0</td>
          <td>9525.0</td>
          <td>4.7</td>
          <td>244.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>0</td>
          <td>4.892100</td>
          <td>752.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>10400.0</td>
          <td>10400.0</td>
          <td>36</td>
          <td>6.99</td>
          <td>321.08</td>
          <td>0</td>
          <td>2</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>...</td>
          <td>9536.0</td>
          <td>7599.0</td>
          <td>41.5</td>
          <td>290.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>9.0</td>
          <td>1</td>
          <td>4.763435</td>
          <td>712.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21425.0</td>
          <td>21425.0</td>
          <td>60</td>
          <td>15.59</td>
          <td>516.36</td>
          <td>3</td>
          <td>15</td>
          <td>5</td>
          <td>1</td>
          <td>1</td>
          <td>...</td>
          <td>4232.0</td>
          <td>324.0</td>
          <td>97.8</td>
          <td>136.0</td>
          <td>7.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>0</td>
          <td>4.804827</td>
          <td>687.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7650.0</td>
          <td>7650.0</td>
          <td>36</td>
          <td>13.66</td>
          <td>260.20</td>
          <td>2</td>
          <td>12</td>
          <td>5</td>
          <td>1</td>
          <td>2</td>
          <td>...</td>
          <td>5857.0</td>
          <td>332.0</td>
          <td>93.2</td>
          <td>148.0</td>
          <td>8.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>1</td>
          <td>4.698979</td>
          <td>687.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>9600.0</td>
          <td>9600.0</td>
          <td>36</td>
          <td>13.66</td>
          <td>326.53</td>
          <td>2</td>
          <td>12</td>
          <td>5</td>
          <td>1</td>
          <td>2</td>
          <td>...</td>
          <td>3214.0</td>
          <td>6494.0</td>
          <td>69.2</td>
          <td>265.0</td>
          <td>23.0</td>
          <td>0.0</td>
          <td>7.0</td>
          <td>0</td>
          <td>4.838855</td>
          <td>682.0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 29 columns</p>
    </div>



.. code:: ipython3

    dataset.shape




.. parsed-literal::

    (814986, 29)



 ## 4.3. Sampling Data

.. code:: ipython3

    loanstatus_0 = dataset[dataset["charged_off"]==0]
    loanstatus_1 = dataset[dataset["charged_off"]==1]
    subset_of_loanstatus_0 = loanstatus_0.sample(n=5500)
    subset_of_loanstatus_1 = loanstatus_1.sample(n=5500)
    dataset = pd.concat([subset_of_loanstatus_1, subset_of_loanstatus_0])
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    print("Current shape of dataset :",dataset.shape)
    dataset.head()


.. parsed-literal::

    Current shape of dataset : (11000, 29)




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
          <th>loan_amnt</th>
          <th>funded_amnt</th>
          <th>term</th>
          <th>int_rate</th>
          <th>installment</th>
          <th>grade</th>
          <th>sub_grade</th>
          <th>home_ownership</th>
          <th>verification_status</th>
          <th>purpose</th>
          <th>...</th>
          <th>avg_cur_bal</th>
          <th>bc_open_to_buy</th>
          <th>bc_util</th>
          <th>mo_sin_old_rev_tl_op</th>
          <th>mo_sin_rcnt_rev_tl_op</th>
          <th>mort_acc</th>
          <th>num_actv_rev_tl</th>
          <th>charged_off</th>
          <th>log_annual_inc</th>
          <th>fico_score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5000.0</td>
          <td>5000.0</td>
          <td>36</td>
          <td>5.42</td>
          <td>150.80</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>4</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1</td>
          <td>4.698979</td>
          <td>777.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>6000.0</td>
          <td>6000.0</td>
          <td>36</td>
          <td>14.46</td>
          <td>206.41</td>
          <td>2</td>
          <td>13</td>
          <td>1</td>
          <td>1</td>
          <td>4</td>
          <td>...</td>
          <td>31847.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>106.0</td>
          <td>6.0</td>
          <td>2.0</td>
          <td>3.0</td>
          <td>0</td>
          <td>4.863329</td>
          <td>662.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>10000.0</td>
          <td>10000.0</td>
          <td>36</td>
          <td>11.99</td>
          <td>332.10</td>
          <td>2</td>
          <td>10</td>
          <td>4</td>
          <td>1</td>
          <td>2</td>
          <td>...</td>
          <td>6058.0</td>
          <td>4935.0</td>
          <td>56.3</td>
          <td>65.0</td>
          <td>10.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>0</td>
          <td>4.447174</td>
          <td>672.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30000.0</td>
          <td>30000.0</td>
          <td>60</td>
          <td>23.50</td>
          <td>854.36</td>
          <td>5</td>
          <td>25</td>
          <td>1</td>
          <td>1</td>
          <td>2</td>
          <td>...</td>
          <td>8282.0</td>
          <td>24329.0</td>
          <td>55.9</td>
          <td>393.0</td>
          <td>116.0</td>
          <td>4.0</td>
          <td>3.0</td>
          <td>1</td>
          <td>4.977728</td>
          <td>697.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>15000.0</td>
          <td>15000.0</td>
          <td>60</td>
          <td>13.98</td>
          <td>348.87</td>
          <td>2</td>
          <td>12</td>
          <td>1</td>
          <td>2</td>
          <td>2</td>
          <td>...</td>
          <td>26643.0</td>
          <td>4146.0</td>
          <td>62.6</td>
          <td>180.0</td>
          <td>1.0</td>
          <td>3.0</td>
          <td>6.0</td>
          <td>0</td>
          <td>4.851264</td>
          <td>707.0</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 29 columns</p>
    </div>



.. code:: ipython3

    #Filling the NAs with the mean of the column.
    dataset.fillna(dataset.mean(),inplace = True)

 # 5. Evaluate Algorithms and Models

 ## 5.1. Train Test Split

.. code:: ipython3

    # split out validation dataset for the end
    Y= dataset["charged_off"]
    X = dataset.loc[:, dataset.columns != 'charged_off']
    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

.. code:: ipython3

    # dataset_temp2=dataset_temp.dropna(axis=0)
    # Y_total= dataset_temp2["charged_off"]
    # X_total = dataset_temp2.loc[:, dataset.columns != 'charged_off']
    # X_dummy, X_validation, Y_dummy, Y_validation = train_test_split(X_total, Y_total, test_size=validation_size)

.. code:: ipython3

    dataset['charged_off'].value_counts()




.. parsed-literal::

    1    5500
    0    5500
    Name: charged_off, dtype: int64



 ## 5.2. Test Options and Evaluation Metrics

.. code:: ipython3

    # test options for classification
    num_folds = 10
    seed = 7
    #scoring = 'accuracy'
    #scoring ='precision'
    #scoring ='recall'
    scoring = 'roc_auc'

 ## 5.3. Compare Models and Algorithms

Classification Models
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # spot check the algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #Neural Network
    models.append(('NN', MLPClassifier()))
    #Ensable Models
    # Boosting methods
    models.append(('AB', AdaBoostClassifier()))
    models.append(('GBM', GradientBoostingClassifier()))
    # Bagging methods
    models.append(('RF', RandomForestClassifier()))
    models.append(('ET', ExtraTreesClassifier()))

K-folds cross validation
~~~~~~~~~~~~~~~~~~~~~~~~

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

    LR: 0.919850 (0.010151)
    LDA: 0.918040 (0.009574)
    KNN: 0.870372 (0.012865)
    CART: 0.831503 (0.014110)
    NB: 0.917332 (0.007952)
    NN: 0.911734 (0.017527)
    AB: 0.944973 (0.005927)
    GBM: 0.952306 (0.006132)
    RF: 0.948115 (0.006236)
    ET: 0.939516 (0.006260)


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



.. image:: output_92_0.png


 # 6. Model Tuning and Grid Search

Given that the GBM is the best model, Grid Search is performed on GBM in
this step.

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
    n_estimators = [20,180]
    max_depth= [3,5]
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    model = GradientBoostingClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)

    #Print Results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    ranks = grid_result.cv_results_['rank_test_score']
    for mean, stdev, param, rank in zip(means, stds, params, ranks):
        print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))


.. parsed-literal::

    Best: 0.955684 using {'max_depth': 5, 'n_estimators': 180}
    #4 0.941023 (0.006713) with: {'max_depth': 3, 'n_estimators': 20}
    #2 0.955217 (0.006234) with: {'max_depth': 3, 'n_estimators': 180}
    #3 0.948795 (0.006683) with: {'max_depth': 5, 'n_estimators': 20}
    #1 0.955684 (0.007695) with: {'max_depth': 5, 'n_estimators': 180}


 # 7. Finalise the Model

Looking at the details above GBM might be worthy of further study, but
for now SVM shows a lot of promise as a low complexity and stable model
for this problem.

Finalize Model with best parameters found during tuning step.

 ## 7.1. Results on the Test Dataset

.. code:: ipython3

    # prepare model
    model = GradientBoostingClassifier(max_depth= 5, n_estimators= 180)
    model.fit(X_train, Y_train)




.. parsed-literal::

    GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=5,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=180,
                               n_iter_no_change=None, presort='deprecated',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)



.. code:: ipython3

    # estimate accuracy on validation set
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


.. parsed-literal::

    0.8909090909090909
    [[ 929  185]
     [  55 1031]]
                  precision    recall  f1-score   support

               0       0.94      0.83      0.89      1114
               1       0.85      0.95      0.90      1086

        accuracy                           0.89      2200
       macro avg       0.90      0.89      0.89      2200
    weighted avg       0.90      0.89      0.89      2200



.. code:: ipython3

    df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation), index = np.unique(Y_validation))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font sizes




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x2569ea9a208>




.. image:: output_101_1.png


 ## 7.2. Variable Intuition/Feature Importance Looking at the details
above GBM might be worthy of further study. Let us look into the Feature
Importance of the GBM model

.. code:: ipython3

    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    pyplot.show()


.. parsed-literal::

    [6.84986527e-03 5.71433689e-03 2.72277382e-02 1.59752215e-02
     2.81338346e-02 9.58442384e-03 5.44634871e-02 4.32347990e-04
     4.07216329e-03 2.75786223e-03 5.61551419e-03 1.26744468e-02
     7.71521749e-03 4.86345328e-03 7.90633550e-03 3.58958849e-03
     7.26846409e-01 0.00000000e+00 1.09532061e-02 1.51860077e-02
     6.52805217e-03 4.70847054e-03 8.09053091e-03 7.63984237e-03
     2.73216743e-03 3.50115469e-03 9.93457709e-03 6.30374494e-03]



.. image:: output_103_1.png


**Conclusion**:

We showed that data preparation is one of the most important steps. We
addressed that by performing feature elimination by using different
techniques such as subjec‐ tive judgement, correlation, visualization
and the data quality of the feature. We illustrated that there can be
different ways of handling and analyzing the categorical data and
converting categorical data into model usable format.

Finally, we analyzed the feature importance and found that results of
the case study are quite intuitive.
