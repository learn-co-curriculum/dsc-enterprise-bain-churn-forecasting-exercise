
# Churn forecasting and management

https://www.kaggle.com/pangkw/telco-churn/version/3

__Business case__: Our client is Triple Telco and they provide triple service communication services to consumers (landline phone, fast internet and TV service). Up until a few months ago they were the only triple play provider in their area. However, new provider entered their market and they've seen a dramatic increase in customers who "churn" (i.e. leave).

They hired Bain to help identify drivers of churn and recommend actions to prevent it.

__Note: This notebook will not be shared with trainees__

# Sprint 1

* Load the necessary libraries
* Load the data
* Explore data – what types, any obvious issues
* Run mechanical feature engineering (longest part)
* Split data
* Run logistic regression
* Explore coefficients
* Deep dive on the most predictive data
* Clean-up data (if necessary)
* Re-run logistic regression (if necessary)




## Load all the necessary libraries


```javascript
%%javascript
// Ensure our output is sufficiently large
IPython.OutputArea.auto_scroll_threshold = 9999;
```


    <IPython.core.display.Javascript object>



```python
# Emnsure all charts plotted automatically
%matplotlib inline

import warnings
import numpy as np
import pandas as pd

import sklearn
import sklearn.linear_model

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn import preprocessing

from sklearn_pandas import DataFrameMapper

from xgboost import XGBClassifier

from IPython.display import display, Markdown

import matplotlib.pyplot as plt
import seaborn as sns

# Don't truncate dataframe displays columwise
pd.set_option('display.max_columns', None)  
```

## Engineer Data: Load the data

You load the CSV data into a Pandas object. This is a common Python library to work with data in row/column format, like .csv files.

Print out top 10 rows just to get a feel for the data


```python


df = pd.read_csv('telco_churn_dataset.csv', sep=',', header=0)
display(df.head(10))
display(df.shape)
```


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
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>MaritalStatus</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>InternationalPlan</th>
      <th>VoiceMailPlan</th>
      <th>NumbervMailMessages</th>
      <th>TotalDayMinutes</th>
      <th>TotalDayCalls</th>
      <th>TotalEveMinutes</th>
      <th>TotalEveCalls</th>
      <th>TotalNightMinutes</th>
      <th>TotalNightCalls</th>
      <th>TotalIntlMinutes</th>
      <th>TotalIntlCalls</th>
      <th>CustomerServiceCalls</th>
      <th>TotalCall</th>
      <th>TotalHighBandwidthMinutes</th>
      <th>TotalHighLatencyMinutes</th>
      <th>TotalRevenue</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002-ORFBO</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>9</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>168.8</td>
      <td>137</td>
      <td>241.4</td>
      <td>107</td>
      <td>204.8</td>
      <td>106</td>
      <td>15.5</td>
      <td>4</td>
      <td>0</td>
      <td>354</td>
      <td>705</td>
      <td>119</td>
      <td>593.3</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0004-TLHLJ</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>122.2</td>
      <td>112</td>
      <td>131.7</td>
      <td>94</td>
      <td>169.5</td>
      <td>106</td>
      <td>10.3</td>
      <td>9</td>
      <td>5</td>
      <td>326</td>
      <td>292</td>
      <td>75</td>
      <td>280.9</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0013-MHZWF</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Yes</td>
      <td>9</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>No</td>
      <td>Yes</td>
      <td>36</td>
      <td>178.7</td>
      <td>134</td>
      <td>178.6</td>
      <td>102</td>
      <td>126.8</td>
      <td>82</td>
      <td>8.0</td>
      <td>4</td>
      <td>2</td>
      <td>324</td>
      <td>1840</td>
      <td>257</td>
      <td>571.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0013-SMEOE</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>71</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>190.2</td>
      <td>68</td>
      <td>262.2</td>
      <td>64</td>
      <td>130.0</td>
      <td>92</td>
      <td>8.8</td>
      <td>4</td>
      <td>0</td>
      <td>228</td>
      <td>1389</td>
      <td>180</td>
      <td>7904.3</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0015-UOCOJ</td>
      <td>Female</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>7</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>67.7</td>
      <td>68</td>
      <td>195.7</td>
      <td>86</td>
      <td>236.5</td>
      <td>137</td>
      <td>12.0</td>
      <td>2</td>
      <td>1</td>
      <td>294</td>
      <td>170</td>
      <td>1</td>
      <td>340.4</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0018-NYROU</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>5</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>174.3</td>
      <td>95</td>
      <td>186.6</td>
      <td>128</td>
      <td>258.2</td>
      <td>105</td>
      <td>12.9</td>
      <td>5</td>
      <td>3</td>
      <td>336</td>
      <td>112</td>
      <td>17</td>
      <td>351.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0021-IKXGC</td>
      <td>Female</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>111.9</td>
      <td>55</td>
      <td>223.0</td>
      <td>124</td>
      <td>243.2</td>
      <td>81</td>
      <td>10.0</td>
      <td>7</td>
      <td>3</td>
      <td>270</td>
      <td>47</td>
      <td>9</td>
      <td>72.1</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0022-TCJCI</td>
      <td>Male</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>303.2</td>
      <td>133</td>
      <td>170.5</td>
      <td>86</td>
      <td>227.6</td>
      <td>80</td>
      <td>11.5</td>
      <td>3</td>
      <td>0</td>
      <td>302</td>
      <td>450</td>
      <td>76</td>
      <td>2791.5</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0030-FNXPP</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>60.4</td>
      <td>158</td>
      <td>306.2</td>
      <td>120</td>
      <td>123.9</td>
      <td>46</td>
      <td>12.4</td>
      <td>3</td>
      <td>1</td>
      <td>328</td>
      <td>0</td>
      <td>0</td>
      <td>57.2</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0031-PVLZI</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>4</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>No internet service</td>
      <td>Month-to-month</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>321.3</td>
      <td>99</td>
      <td>167.9</td>
      <td>93</td>
      <td>193.6</td>
      <td>106</td>
      <td>8.0</td>
      <td>4</td>
      <td>1</td>
      <td>303</td>
      <td>0</td>
      <td>0</td>
      <td>76.4</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



    (3333, 35)


__You should see:__

* 3333 rows of data across all 35 columns (different variables)
* lots of rows have text data that needs to be converted into numerical data, ideally one-hot encoded


## Explore the data: 

* Check  for nulls
* Check for outliers on data that is numerical

### Check for nulls and drop if any


```python
# #First replace all empty strings with NAN
df.replace(' ', np.nan,  inplace=True)

# Any null values
df.isnull().sum(axis = 0)

display(df.shape)
df = df.dropna()
display(df.shape)


```


    (3333, 35)



    (3328, 35)


__What you should see__:

* We have only 5 null values
 * Think OK to drop
 * In real world, you'd want to document this AND maybe even try to impute the missing number (since only missing Total Reveneu)


### Check for outliers in numerical data 


```python
display(df.dtypes)

num_list = list(df.select_dtypes(include=['int64','float64']).columns)
for column in num_list:
    plt.hist(df[column].dropna())
    plt.title(column)
    plt.show()

```


    customerID                    object
    gender                        object
    SeniorCitizen                  int64
    MaritalStatus                 object
    Dependents                    object
    tenure                         int64
    PhoneService                  object
    MultipleLines                 object
    InternetService               object
    OnlineSecurity                object
    OnlineBackup                  object
    DeviceProtection              object
    TechSupport                   object
    StreamingTV                   object
    StreamingMovies               object
    Contract                      object
    PaperlessBilling              object
    PaymentMethod                 object
    InternationalPlan             object
    VoiceMailPlan                 object
    NumbervMailMessages            int64
    TotalDayMinutes              float64
    TotalDayCalls                  int64
    TotalEveMinutes              float64
    TotalEveCalls                  int64
    TotalNightMinutes            float64
    TotalNightCalls                int64
    TotalIntlMinutes             float64
    TotalIntlCalls                 int64
    CustomerServiceCalls           int64
    TotalCall                      int64
    TotalHighBandwidthMinutes      int64
    TotalHighLatencyMinutes        int64
    TotalRevenue                  object
    Churn                         object
    dtype: object



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)



![png](output_13_4.png)



![png](output_13_5.png)



![png](output_13_6.png)



![png](output_13_7.png)



![png](output_13_8.png)



![png](output_13_9.png)



![png](output_13_10.png)



![png](output_13_11.png)



![png](output_13_12.png)



![png](output_13_13.png)



![png](output_13_14.png)



![png](output_13_15.png)


__What you should see__:
* Nothing too suspicious, think can assume data is clean
* Revenue is not a float or int - we should fix that


```python
df['TotalRevenue']= pd.to_numeric(df['TotalRevenue'], errors='coerce').fillna(0, downcast='infer')
```

## Mechanical Feature Engineering


```python
# We use capitals for constant variables
TARGET = 'Churn'
ID =  'customerID'
FEATURES = df.columns.tolist()

# Remove customer id - it can't possibly have any predictive power
FEATURES.remove(ID)
#remove target from list of features
FEATURES.remove(TARGET)



encoders = ['gender', 'SeniorCitizen', 'MaritalStatus','Dependents', 'PhoneService',
            'MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternationalPlan',
            'VoiceMailPlan']

scalars = ['tenure', 'NumbervMailMessages','TotalDayMinutes','TotalDayCalls',
           'TotalEveMinutes', 'TotalEveCalls', 'TotalNightMinutes', 'TotalNightCalls',
           'TotalIntlMinutes','TotalIntlCalls', 'CustomerServiceCalls', 'TotalCall',
           'TotalHighBandwidthMinutes', 'TotalHighLatencyMinutes', 'TotalRevenue']



preprocessing_steps = ([(encoder, preprocessing.LabelBinarizer()) for encoder in encoders] 
                       + [([scalar], preprocessing.StandardScaler()) for scalar in scalars] )

mapper_features = DataFrameMapper(preprocessing_steps)

np_transformed_features = mapper_features.fit_transform(df[FEATURES].copy())
df_transformed_features = pd.DataFrame(data = np_transformed_features, columns = mapper_features.transformed_names_)

mapper_target = DataFrameMapper([(['Churn'],preprocessing.LabelBinarizer())])
numpy_transform_target = mapper_target.fit_transform(df[[TARGET]].copy())

df_transformed_target = pd.DataFrame(data = numpy_transform_target, columns = ['Churn'])


```

    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)


## Split data


```python
# The function train_test_split, splits the arguments into two sets.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_transformed_features, df_transformed_target, test_size=0.3, random_state=42, stratify=df_transformed_target)

# Lets verify that our splitting has the same distribution
display(Markdown("__Split in training set__"))
display(pd.Categorical(y_train[TARGET]).describe())
display(Markdown("__Split in test set__"))
display(pd.Categorical(y_test[TARGET]).describe())
```


__Split in training set__



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1872</td>
      <td>0.803778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>457</td>
      <td>0.196222</td>
    </tr>
  </tbody>
</table>
</div>



__Split in test set__



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>803</td>
      <td>0.803804</td>
    </tr>
    <tr>
      <th>1</th>
      <td>196</td>
      <td>0.196196</td>
    </tr>
  </tbody>
</table>
</div>


__What you should see__:

* Frequency of No Chrun (0) and yes chrun (1) should be the same in both training and test set
* There should be about 2,300 data-points in training set and about 1000 in the test set

## Develop model: Train and test base-line model
To train and test the base-line model we look at the type of prediction problem we are dealing with. The target is yes or no churn which is a binary outome.
As such our algorithm needs to be able to predict this. A logistic regression is an algorithm that allows to predict binary variable. 

We will:
* Train the model using linear logistic regression
* Test it against unseen data (i.e. test data)
* Show accuracy on seen data and unseen data

*Note for statistics geeks only: The implementation of the algorithm in Python's scikit-learn is not based on ordinary least square or maximum likelihood estimator, which would be the case in R or SAS. Specifically, it penalizes for complexity (called regularization)


```python
# Run Logistic Regression Training
linear_model = sklearn.linear_model.LogisticRegression(random_state=42)
linear_model.fit(X_train, y_train)

display(Markdown('###### Performance on training set'))

y_lr_train = linear_model.predict(X_train)
y_lr_test = linear_model.predict(X_test)
y_lr_test_proba = linear_model.predict_proba(X_test)


# See how well we did
lr_accuracy = sklearn.metrics.accuracy_score(y_train, y_lr_train)
display('Linear model train set accuracy: {:.2f}%'.format(lr_accuracy*100))
display(Markdown('Linear model predicion distribution:'))
display(pd.Categorical(y_lr_test).describe())

lr_accuracy = sklearn.metrics.accuracy_score(y_test, y_lr_test)
display('Linear model test set accuracy: {:.2f}%'.format(lr_accuracy*100))
display(Markdown('Linear model predicion distribution:'))
display(pd.Categorical(y_lr_test).describe())

display(pd.DataFrame(confusion_matrix(y_test, y_lr_test), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_lr_test_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
display(Markdown('\n\nauROC is: {}'.format(roc_auc * 100)))
```

    C:\ML\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:761: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)



###### Performance on training set



    'Linear model train set accuracy: 86.47%'



Linear model predicion distribution:



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>861</td>
      <td>0.861862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138</td>
      <td>0.138138</td>
    </tr>
  </tbody>
</table>
</div>



    'Linear model test set accuracy: 86.59%'



Linear model predicion distribution:



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>861</td>
      <td>0.861862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138</td>
      <td>0.138138</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Predicted Not Churn</th>
      <th>Predicted to Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Not Churn</th>
      <td>765</td>
      <td>38</td>
    </tr>
    <tr>
      <th>Actual Churn</th>
      <td>96</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>





auROC is: 90.32581899509492



```python
#ROC chart
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
```


![png](output_23_0.png)


__What you should see__

* This is pretty good - 86% accuracy (remember overall distribution is 85:15, so a little bit better than a coin-flip)
* AUC is 90 - this is very good for a linear model

* Let's see how these decisions are made

## Explore coefficients

Idea here is to see what the model picked-up to get better understanding why it's predicting churn.

* Explore  intercept
* Explore  coeficients


```python
# get intercept
intercept = linear_model.intercept_

# Get coefficients (same order as features fed into Logistic Regression) and build table
named_coefs = pd.DataFrame({
    'features': df_transformed_features.columns.tolist(),
    'coeficients': linear_model.coef_[0]
}).sort_values('coeficients')

# put it in a nice dataframe and sort from most impact to least, sicne raw print-out is ugly
display("Intercept is {}".format(intercept[0]))
display(named_coefs)
```


    'Intercept is -0.8058042283966471'



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
      <th>features</th>
      <th>coeficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>Contract_Two year</td>
      <td>-1.674577</td>
    </tr>
    <tr>
      <th>36</th>
      <td>VoiceMailPlan</td>
      <td>-1.457715</td>
    </tr>
    <tr>
      <th>37</th>
      <td>tenure</td>
      <td>-0.862083</td>
    </tr>
    <tr>
      <th>24</th>
      <td>StreamingMovies_No</td>
      <td>-0.784354</td>
    </tr>
    <tr>
      <th>6</th>
      <td>InternetService_DSL</td>
      <td>-0.754558</td>
    </tr>
    <tr>
      <th>23</th>
      <td>StreamingTV_Yes</td>
      <td>-0.646839</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Contract_One year</td>
      <td>-0.625388</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DeviceProtection_No</td>
      <td>-0.472732</td>
    </tr>
    <tr>
      <th>25</th>
      <td>StreamingMovies_No internet service</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TechSupport_No internet service</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DeviceProtection_No internet service</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>13</th>
      <td>OnlineBackup_No internet service</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>22</th>
      <td>StreamingTV_No internet service</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>10</th>
      <td>OnlineSecurity_No internet service</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>8</th>
      <td>InternetService_No</td>
      <td>-0.400562</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PaymentMethod_Mailed check</td>
      <td>-0.306224</td>
    </tr>
    <tr>
      <th>12</th>
      <td>OnlineBackup_No</td>
      <td>-0.286769</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TechSupport_No</td>
      <td>-0.253757</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PaymentMethod_Bank transfer (automatic)</td>
      <td>-0.249780</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PaymentMethod_Credit card (automatic)</td>
      <td>-0.242539</td>
    </tr>
    <tr>
      <th>9</th>
      <td>OnlineSecurity_No</td>
      <td>-0.232556</td>
    </tr>
    <tr>
      <th>11</th>
      <td>OnlineSecurity_Yes</td>
      <td>-0.172686</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TechSupport_Yes</td>
      <td>-0.151484</td>
    </tr>
    <tr>
      <th>0</th>
      <td>gender</td>
      <td>-0.137794</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TotalHighBandwidthMinutes</td>
      <td>-0.132308</td>
    </tr>
    <tr>
      <th>14</th>
      <td>OnlineBackup_Yes</td>
      <td>-0.118473</td>
    </tr>
    <tr>
      <th>46</th>
      <td>TotalIntlCalls</td>
      <td>-0.105559</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MultipleLines</td>
      <td>-0.102840</td>
    </tr>
    <tr>
      <th>42</th>
      <td>TotalEveCalls</td>
      <td>-0.075236</td>
    </tr>
    <tr>
      <th>44</th>
      <td>TotalNightCalls</td>
      <td>-0.013451</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PaymentMethod_Electronic check</td>
      <td>-0.007261</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhoneService</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TotalCall</td>
      <td>0.025719</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SeniorCitizen</td>
      <td>0.026685</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dependents</td>
      <td>0.042888</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DeviceProtection_Yes</td>
      <td>0.067491</td>
    </tr>
    <tr>
      <th>40</th>
      <td>TotalDayCalls</td>
      <td>0.114031</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NumbervMailMessages</td>
      <td>0.178559</td>
    </tr>
    <tr>
      <th>41</th>
      <td>TotalEveMinutes</td>
      <td>0.189349</td>
    </tr>
    <tr>
      <th>45</th>
      <td>TotalIntlMinutes</td>
      <td>0.194472</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MaritalStatus</td>
      <td>0.196800</td>
    </tr>
    <tr>
      <th>43</th>
      <td>TotalNightMinutes</td>
      <td>0.206515</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PaperlessBilling</td>
      <td>0.240318</td>
    </tr>
    <tr>
      <th>21</th>
      <td>StreamingTV_No</td>
      <td>0.241598</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TotalRevenue</td>
      <td>0.273972</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TotalHighLatencyMinutes</td>
      <td>0.292863</td>
    </tr>
    <tr>
      <th>7</th>
      <td>InternetService_Fiber optic</td>
      <td>0.349317</td>
    </tr>
    <tr>
      <th>26</th>
      <td>StreamingMovies_Yes</td>
      <td>0.379112</td>
    </tr>
    <tr>
      <th>39</th>
      <td>TotalDayMinutes</td>
      <td>0.452253</td>
    </tr>
    <tr>
      <th>47</th>
      <td>CustomerServiceCalls</td>
      <td>0.471088</td>
    </tr>
    <tr>
      <th>35</th>
      <td>InternationalPlan</td>
      <td>1.343506</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Contract_Month-to-month</td>
      <td>1.494161</td>
    </tr>
  </tbody>
</table>
</div>


__What you should see__

* list of weights on attributes that drive folks to churn (+ number) or not churn(- number)
* Top five things making folks more likely to churn:
 * International plan - a bit odd - maybe we overcharged them a ton?
 * Contract_Month-to-month - makes a ton of sense, folks are ready to leave
 * Total revenue - seems customers who spend a ton, maybe competitors are cheaper
 * Customer service calls - makes a ton of sense
 * TotalDayMinutes - folks who talk a lot - that's bad - these are probably best customers
 
    
* Top five attributes contributing to not churning:
 * 2 year contract - makes sense - they can't leave without penalty
 * voicemailplan - folks who have voicemail with us - maybe they really like this feature
 * Multiple lines - maybe our competitors don't offer this feature?
 * tenure - folks who are with us for a long time, like to stay with us
 * Contract - one year

# Sprint 2

* Then let's run XGBoost model

* Let's do a bit of feature engineering:
 * Maybe those who have lots of latency when they watch online tv/movies (i.e. share of high bandwidth mintues that are high latency)
 * Those who overpay for services
 


## First - let's run it as XGBoost


```python
XG_model = XGBClassifier()
XG_model.fit(X_train, y_train)

y_predict = XG_model.predict(X_test)
y_predict_proba = XG_model.predict_proba(X_test)

xg_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
display('XGBoost model test set accuracy: {:.4f}'.format(xg_accuracy))
display('XGBoost model predicion distribution')
display(pd.Categorical(y_predict).describe())

display(pd.DataFrame(confusion_matrix(y_test, y_predict), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
display("auROC is: {}".format(roc_auc * 100))
```

    C:\ML\anaconda3\lib\site-packages\sklearn\preprocessing\label.py:219: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    C:\ML\anaconda3\lib\site-packages\sklearn\preprocessing\label.py:252: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)



    'XGBoost model test set accuracy: 0.9119'



    'XGBoost model predicion distribution'



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>843</td>
      <td>0.843844</td>
    </tr>
    <tr>
      <th>1</th>
      <td>156</td>
      <td>0.156156</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Predicted Not Churn</th>
      <th>Predicted to Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Not Churn</th>
      <td>779</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Actual Churn</th>
      <td>64</td>
      <td>132</td>
    </tr>
  </tbody>
</table>
</div>



    'auROC is: 94.80074719800747'


___What you should see:___

* Impressive improvment
 * Accuracy up to 91% and auRoc to 95!
 
__What is next__: Let's see feature importnace 


```python
#ROC chart
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
```


![png](output_32_0.png)


### Let's understand what's going on with features


```python
pd.DataFrame({'features': df_transformed_features.columns.tolist(), 'importance': XG_model.feature_importances_}).sort_values('importance', ascending=False)
```




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
      <th>features</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>TotalDayMinutes</td>
      <td>0.188474</td>
    </tr>
    <tr>
      <th>41</th>
      <td>TotalEveMinutes</td>
      <td>0.087227</td>
    </tr>
    <tr>
      <th>47</th>
      <td>CustomerServiceCalls</td>
      <td>0.074766</td>
    </tr>
    <tr>
      <th>45</th>
      <td>TotalIntlMinutes</td>
      <td>0.060748</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TotalHighLatencyMinutes</td>
      <td>0.057632</td>
    </tr>
    <tr>
      <th>35</th>
      <td>InternationalPlan</td>
      <td>0.052960</td>
    </tr>
    <tr>
      <th>43</th>
      <td>TotalNightMinutes</td>
      <td>0.049844</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Contract_Month-to-month</td>
      <td>0.045171</td>
    </tr>
    <tr>
      <th>36</th>
      <td>VoiceMailPlan</td>
      <td>0.042056</td>
    </tr>
    <tr>
      <th>37</th>
      <td>tenure</td>
      <td>0.035826</td>
    </tr>
    <tr>
      <th>26</th>
      <td>StreamingMovies_Yes</td>
      <td>0.031153</td>
    </tr>
    <tr>
      <th>46</th>
      <td>TotalIntlCalls</td>
      <td>0.028037</td>
    </tr>
    <tr>
      <th>7</th>
      <td>InternetService_Fiber optic</td>
      <td>0.028037</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TotalHighBandwidthMinutes</td>
      <td>0.026480</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TotalRevenue</td>
      <td>0.023364</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TechSupport_No</td>
      <td>0.021807</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NumbervMailMessages</td>
      <td>0.017134</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DeviceProtection_Yes</td>
      <td>0.017134</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TotalCall</td>
      <td>0.017134</td>
    </tr>
    <tr>
      <th>44</th>
      <td>TotalNightCalls</td>
      <td>0.015576</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Contract_Two year</td>
      <td>0.012461</td>
    </tr>
    <tr>
      <th>21</th>
      <td>StreamingTV_No</td>
      <td>0.010903</td>
    </tr>
    <tr>
      <th>42</th>
      <td>TotalEveCalls</td>
      <td>0.009346</td>
    </tr>
    <tr>
      <th>40</th>
      <td>TotalDayCalls</td>
      <td>0.007788</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PaymentMethod_Electronic check</td>
      <td>0.007788</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Contract_One year</td>
      <td>0.006231</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PaperlessBilling</td>
      <td>0.006231</td>
    </tr>
    <tr>
      <th>8</th>
      <td>InternetService_No</td>
      <td>0.003115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MaritalStatus</td>
      <td>0.003115</td>
    </tr>
    <tr>
      <th>14</th>
      <td>OnlineBackup_Yes</td>
      <td>0.003115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SeniorCitizen</td>
      <td>0.001558</td>
    </tr>
    <tr>
      <th>24</th>
      <td>StreamingMovies_No</td>
      <td>0.001558</td>
    </tr>
    <tr>
      <th>6</th>
      <td>InternetService_DSL</td>
      <td>0.001558</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PaymentMethod_Credit card (automatic)</td>
      <td>0.001558</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dependents</td>
      <td>0.001558</td>
    </tr>
    <tr>
      <th>12</th>
      <td>OnlineBackup_No</td>
      <td>0.001558</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MultipleLines</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhoneService</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>OnlineSecurity_No</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DeviceProtection_No</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>OnlineSecurity_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>OnlineSecurity_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>OnlineBackup_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PaymentMethod_Mailed check</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DeviceProtection_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PaymentMethod_Bank transfer (automatic)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>StreamingMovies_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>StreamingTV_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>StreamingTV_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TechSupport_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TechSupport_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>gender</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



___What you should see:___

* Seems features around time on the phone are dominating 
* Latency minutes drive some dissatisfaction too
* Contract and tenure too


##  Feature Engineering 

* first let's create a features
* then explore it to see if it may be a signal
* then re-run model with it

Features:
* Latency as share of traffic
* Dollars paid  per service



### First up - Latency minutes - is there a signal here?

* What we care about here is what proportion of time when download speeds are high (e.g. movie watching) also experiences high latency - to do that we'll need to divide high latency minutes by high bandwidth minutes
* Since we scalarized these in our feature-engineered one, we need to work on original dataframe


```python
df_latency = pd.DataFrame()
df_latency['Churn'] = df['Churn'].copy()
df_latency['TotalHighBandwidthMinutes'] = df['TotalHighBandwidthMinutes'].copy()
df_latency['TotalHighLatencyMinutes'] = df['TotalHighLatencyMinutes'].copy()
df_latency['Latency_share'] = df_latency['TotalHighLatencyMinutes']/df_latency['TotalHighBandwidthMinutes']
df_latency['Latency_share'] = df_latency['Latency_share'].fillna(0)

plt.hist(df_latency['Latency_share'], normed=True, bins=10)

latency_bins = [df_latency['Latency_share'].min(), .1, .2, .3, .4, df_latency['Latency_share'].max()]
df_latency['LatencyShareBin'] = pd.cut(df_latency['Latency_share'], latency_bins,include_lowest=True).astype(str)

display(pd.crosstab(df_latency['LatencyShareBin'],df_latency['Churn'],normalize='index',margins=True))

       
```

    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



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
      <th>Churn</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>LatencyShareBin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(-0.001, 0.1]</th>
      <td>0.876757</td>
      <td>0.123243</td>
    </tr>
    <tr>
      <th>(0.1, 0.2]</th>
      <td>0.714491</td>
      <td>0.285509</td>
    </tr>
    <tr>
      <th>(0.2, 0.3]</th>
      <td>0.748641</td>
      <td>0.251359</td>
    </tr>
    <tr>
      <th>(0.3, 0.4]</th>
      <td>0.111111</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>(0.4, 0.502]</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>All</th>
      <td>0.803786</td>
      <td>0.196214</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_38_2.png)


___What you should see___:

* Looks like most people have latency <5% of total traffic
* For very small group, it goes up above 30%
* Churn doubles for those with latency over 10% of traffi
* Churn is nearly certain for those whose latency is over 30% of traffic

__Takeaway - this a very good feature to add into the model__


 

### Next-up, are people over-paying:

* To do that - we need to: 
 * figure out how much are people paying per month (total rev/tenure)
 * figure out how many services they have (Count phone, interent, tv)
 * figure out what is month charge/service look like
 * Remember, need to look at month-to-month folks only


```python
df_overpayment = df.copy()


# Apparently TotalRevnue cannot be downacasted to flaot by auto-magic so forcing it here
df['TotalRevenue']= pd.to_numeric(df['TotalRevenue'], errors='coerce').fillna(0, downcast='infer')

df_overpayment['AvgRevPerMonth'] = df['TotalRevenue'] / df['tenure']
# df_overpayment['num_services'] = df_transformed_features['PhoneService'] + df_transformed_features['InternetService_DSL'] + df_transformed_features['InternetService_Fiber optic'] + df_transformed_features['StreamingTV_Yes'] + 1
df_overpayment['num_services'] = df['PhoneService'].apply(lambda x: 0 if x == "No" else 1)
df_overpayment['num_services'] = df_overpayment['num_services'] + df['InternetService'].apply(lambda x: 0 if x == "No" else 1)
df_overpayment['num_services'] = df_overpayment['num_services'] + df['StreamingTV'].apply(lambda x: 1 if x == "Yes" else 0)

display(pd.crosstab(df_overpayment['num_services'],df_overpayment['Churn'],normalize='index',margins=True))
display(pd.crosstab(df_overpayment['num_services'],df_overpayment['Churn'],margins=True))



df_overpayment['AvgRevPerServicePerMonth'] = df_overpayment['AvgRevPerMonth'] / df_overpayment['num_services']
df_overpayment['AvgRevPerServicePerMonth'] = df_overpayment['AvgRevPerServicePerMonth'].fillna(0)

plt.hist(df_overpayment[(df_overpayment['Churn'] == 'Yes') & (df_overpayment['num_services'] == 1) & (df_overpayment['Contract'] == "Month-to-month")]['AvgRevPerServicePerMonth'].dropna(), bins=20, alpha=0.5, label='Churners', normed = True)
plt.hist(df_overpayment[(df_overpayment['Churn'] == 'No') & (df_overpayment['num_services'] == 1) & (df_overpayment['Contract'] == "Month-to-month")]['AvgRevPerServicePerMonth'].dropna(), bins=20, alpha=0.5, label='Stayers',normed = True)
plt.legend(loc='upper right')
plt.title('1 service')
plt.show()


plt.hist(df_overpayment[(df_overpayment['Churn'] == 'Yes') & (df_overpayment['num_services'] == 2) & (df_overpayment['Contract'] == "Month-to-month")]['AvgRevPerServicePerMonth'].dropna(), bins=20, alpha=0.5, label='Churners', normed = True)
plt.hist(df_overpayment[(df_overpayment['Churn'] == 'No') & (df_overpayment['num_services'] == 2) & (df_overpayment['Contract'] == "Month-to-month")]['AvgRevPerServicePerMonth'].dropna(), bins=20, alpha=0.5, label='Stayers',normed = True)
plt.legend(loc='upper right')
plt.title('2 services')
plt.show()


plt.hist(df_overpayment[(df_overpayment['Churn'] == 'Yes') & (df_overpayment['num_services'] == 3) & (df_overpayment['Contract'] == "Month-to-month")]['AvgRevPerServicePerMonth'].dropna(), bins=20, alpha=0.5, label='Churners', normed = True)
plt.hist(df_overpayment[(df_overpayment['Churn'] == 'No') & (df_overpayment['num_services'] == 3) & (df_overpayment['Contract'] == "Month-to-month")]['AvgRevPerServicePerMonth'].dropna(), bins=20, alpha=0.5, label='Stayers',normed = True)
plt.legend(loc='upper right')
plt.title('3 services')
plt.show()


```


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
      <th>Churn</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>num_services</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.949787</td>
      <td>0.050213</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.692490</td>
      <td>0.307510</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.769144</td>
      <td>0.230856</td>
    </tr>
    <tr>
      <th>All</th>
      <td>0.803786</td>
      <td>0.196214</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Churn</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>num_services</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1116</td>
      <td>59</td>
      <td>1175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>876</td>
      <td>389</td>
      <td>1265</td>
    </tr>
    <tr>
      <th>3</th>
      <td>683</td>
      <td>205</td>
      <td>888</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2675</td>
      <td>653</td>
      <td>3328</td>
    </tr>
  </tbody>
</table>
</div>


    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](output_41_3.png)


    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](output_41_5.png)


    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")
    C:\ML\anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6521: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      alternative="'density'", removal="3.1")



![png](output_41_7.png)


___What you should see:___

* Looks like folks with multiple services are in fact more likely to churn out, but slightly (crosstab)
* When we plot average dollars paid per service per month we see that regardless of number of services, those with higher price paid are in fact somewhat more likely to churn-out.
* That signal is not very strong, but we can hope that ML algo will be able to make it useful

## Add Features into training/test data set


```python
df_fe = df.copy()

df_fe['Latency_share'] = df_latency['Latency_share'].copy()
df_fe['num_services'] = df_overpayment['num_services'].copy().fillna(0)
df_fe['AvgRevPerServicePerMonth'] = df_overpayment['AvgRevPerServicePerMonth'].copy()

encoders = ['gender', 'SeniorCitizen', 'MaritalStatus','Dependents', 'PhoneService',
            'MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'InternationalPlan',
            'VoiceMailPlan']

scalars = ['tenure', 'NumbervMailMessages','TotalDayMinutes','TotalDayCalls',
           'TotalEveMinutes', 'TotalEveCalls', 'TotalNightMinutes', 'TotalNightCalls',
           'TotalIntlMinutes','TotalIntlCalls', 'CustomerServiceCalls', 'TotalCall',
           'TotalHighBandwidthMinutes', 'TotalHighLatencyMinutes', 'TotalRevenue', 
           'Latency_share', 'num_services','AvgRevPerServicePerMonth']

FEATURES = encoders + scalars

preprocessing_steps = ([(encoder, preprocessing.LabelBinarizer()) for encoder in encoders] 
                       + [([scalar], preprocessing.StandardScaler()) for scalar in scalars] )

mapper_features = DataFrameMapper(preprocessing_steps)
np_transformed_features = mapper_features.fit_transform(df_fe[FEATURES].copy())
df_transformed_features = pd.DataFrame(data = np_transformed_features, columns = mapper_features.transformed_names_)

display(df_transformed_features.head(10))

mapper_target = DataFrameMapper([('Churn',preprocessing.LabelBinarizer())])
numpy_transform_target = mapper_target.fit_transform(df[[TARGET]].copy())
df_transformed_target = pd.DataFrame(data = numpy_transform_target, columns = ['Churn'])
display(df_transformed_target.head(10))

FEATURES = mapper_features.transformed_names_
TARGET = 'Churn'


```

    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    C:\ML\anaconda3\lib\site-packages\sklearn\utils\validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)



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
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>MaritalStatus</th>
      <th>Dependents</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>InternetService_No</th>
      <th>OnlineSecurity_No</th>
      <th>OnlineSecurity_No internet service</th>
      <th>OnlineSecurity_Yes</th>
      <th>OnlineBackup_No</th>
      <th>OnlineBackup_No internet service</th>
      <th>OnlineBackup_Yes</th>
      <th>DeviceProtection_No</th>
      <th>DeviceProtection_No internet service</th>
      <th>DeviceProtection_Yes</th>
      <th>TechSupport_No</th>
      <th>TechSupport_No internet service</th>
      <th>TechSupport_Yes</th>
      <th>StreamingTV_No</th>
      <th>StreamingTV_No internet service</th>
      <th>StreamingTV_Yes</th>
      <th>StreamingMovies_No</th>
      <th>StreamingMovies_No internet service</th>
      <th>StreamingMovies_Yes</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
      <th>InternationalPlan</th>
      <th>VoiceMailPlan</th>
      <th>tenure</th>
      <th>NumbervMailMessages</th>
      <th>TotalDayMinutes</th>
      <th>TotalDayCalls</th>
      <th>TotalEveMinutes</th>
      <th>TotalEveCalls</th>
      <th>TotalNightMinutes</th>
      <th>TotalNightCalls</th>
      <th>TotalIntlMinutes</th>
      <th>TotalIntlCalls</th>
      <th>CustomerServiceCalls</th>
      <th>TotalCall</th>
      <th>TotalHighBandwidthMinutes</th>
      <th>TotalHighLatencyMinutes</th>
      <th>TotalRevenue</th>
      <th>Latency_share</th>
      <th>num_services</th>
      <th>AvgRevPerServicePerMonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.819048</td>
      <td>-0.591443</td>
      <td>-0.201162</td>
      <td>1.822232</td>
      <td>0.797156</td>
      <td>0.345126</td>
      <td>0.077336</td>
      <td>0.300610</td>
      <td>1.884528</td>
      <td>-0.194806</td>
      <td>-1.187957</td>
      <td>1.373011</td>
      <td>0.201463</td>
      <td>0.200284</td>
      <td>-0.562574</td>
      <td>0.601707</td>
      <td>1.387995</td>
      <td>-0.706198</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-1.034094</td>
      <td>-0.591443</td>
      <td>-1.056269</td>
      <td>0.576343</td>
      <td>-1.366832</td>
      <td>-0.307390</td>
      <td>-0.620583</td>
      <td>0.300610</td>
      <td>0.022669</td>
      <td>1.836258</td>
      <td>2.614237</td>
      <td>0.559918</td>
      <td>-0.314672</td>
      <td>-0.082305</td>
      <td>-0.725304</td>
      <td>1.427629</td>
      <td>0.110195</td>
      <td>1.015084</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.819048</td>
      <td>2.039222</td>
      <td>-0.019498</td>
      <td>1.672725</td>
      <td>-0.441663</td>
      <td>0.094158</td>
      <td>-1.464808</td>
      <td>-0.925845</td>
      <td>-0.800846</td>
      <td>-0.194806</td>
      <td>0.332920</td>
      <td>0.501840</td>
      <td>1.619898</td>
      <td>1.086586</td>
      <td>-0.573930</td>
      <td>0.328568</td>
      <td>1.387995</td>
      <td>-0.811978</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.847520</td>
      <td>-0.591443</td>
      <td>0.191526</td>
      <td>-1.616421</td>
      <td>1.207466</td>
      <td>-1.813196</td>
      <td>-1.401540</td>
      <td>-0.414822</td>
      <td>-0.514406</td>
      <td>-0.194806</td>
      <td>-1.187957</td>
      <td>-2.285909</td>
      <td>1.056273</td>
      <td>0.592055</td>
      <td>3.245727</td>
      <td>0.233982</td>
      <td>1.387995</td>
      <td>1.276700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.905066</td>
      <td>-0.591443</td>
      <td>-2.056340</td>
      <td>-1.616421</td>
      <td>-0.104341</td>
      <td>-0.708939</td>
      <td>0.704079</td>
      <td>1.884781</td>
      <td>0.631353</td>
      <td>-1.007232</td>
      <td>-0.427518</td>
      <td>-0.369332</td>
      <td>-0.467139</td>
      <td>-0.557569</td>
      <td>-0.694310</td>
      <td>-0.926346</td>
      <td>0.110195</td>
      <td>-0.399604</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.991085</td>
      <td>-0.591443</td>
      <td>-0.100238</td>
      <td>-0.270861</td>
      <td>-0.283852</td>
      <td>1.399190</td>
      <td>1.133111</td>
      <td>0.249508</td>
      <td>0.953598</td>
      <td>0.211406</td>
      <td>1.093359</td>
      <td>0.850309</td>
      <td>-0.539622</td>
      <td>-0.454809</td>
      <td>-0.688528</td>
      <td>0.442173</td>
      <td>0.110195</td>
      <td>1.019997</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-1.163121</td>
      <td>-0.591443</td>
      <td>-1.245273</td>
      <td>-2.264283</td>
      <td>0.434190</td>
      <td>1.198416</td>
      <td>0.836545</td>
      <td>-0.976947</td>
      <td>-0.084746</td>
      <td>1.023832</td>
      <td>1.093359</td>
      <td>-1.066269</td>
      <td>-0.620854</td>
      <td>-0.506189</td>
      <td>-0.834068</td>
      <td>0.814578</td>
      <td>0.110195</td>
      <td>1.137907</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.729282</td>
      <td>-0.591443</td>
      <td>2.265068</td>
      <td>1.622890</td>
      <td>-0.601447</td>
      <td>-0.708939</td>
      <td>0.528116</td>
      <td>-1.028050</td>
      <td>0.452329</td>
      <td>-0.601019</td>
      <td>-1.187957</td>
      <td>-0.137019</td>
      <td>-0.117216</td>
      <td>-0.075883</td>
      <td>0.582469</td>
      <td>0.602594</td>
      <td>0.110195</td>
      <td>0.478484</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.077103</td>
      <td>-0.591443</td>
      <td>-2.190294</td>
      <td>2.868779</td>
      <td>2.075428</td>
      <td>0.997642</td>
      <td>-1.522144</td>
      <td>-2.765527</td>
      <td>0.774573</td>
      <td>-0.601019</td>
      <td>-0.427518</td>
      <td>0.617996</td>
      <td>-0.679591</td>
      <td>-0.563991</td>
      <td>-0.841829</td>
      <td>-0.981520</td>
      <td>-1.167605</td>
      <td>-1.087102</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.034094</td>
      <td>-0.591443</td>
      <td>2.597202</td>
      <td>-0.071519</td>
      <td>-0.652736</td>
      <td>-0.357584</td>
      <td>-0.144100</td>
      <td>0.300610</td>
      <td>-0.800846</td>
      <td>-0.194806</td>
      <td>-0.427518</td>
      <td>-0.107980</td>
      <td>-0.679591</td>
      <td>-0.563991</td>
      <td>-0.831828</td>
      <td>-0.981520</td>
      <td>-1.167605</td>
      <td>-1.082735</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Split the data


```python
display(Markdown("__Orginal distribution__"))
display(pd.Categorical(df_transformed_target[TARGET]).describe())

# The function train_test_split, splits the arguments into two sets.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_transformed_features[FEATURES], df_transformed_target[TARGET], test_size=0.3, random_state=48879, stratify=df_transformed_target[TARGET])

# Lets verify that our splitting has the same distribution
display(Markdown("__Split in training set__"))
display(pd.Categorical(y_train).describe())
display(Markdown("__Split in test set__"))
display(pd.Categorical(y_test).describe())
```


__Orginal distribution__



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2675</td>
      <td>0.803786</td>
    </tr>
    <tr>
      <th>1</th>
      <td>653</td>
      <td>0.196214</td>
    </tr>
  </tbody>
</table>
</div>



__Split in training set__



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1872</td>
      <td>0.803778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>457</td>
      <td>0.196222</td>
    </tr>
  </tbody>
</table>
</div>



__Split in test set__



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>803</td>
      <td>0.803804</td>
    </tr>
    <tr>
      <th>1</th>
      <td>196</td>
      <td>0.196196</td>
    </tr>
  </tbody>
</table>
</div>


___What you should see___:

* Latency share and AvgRevServicePerMonth are both part of feature set
* We split training and test sets


## Re-run XGBoost


```python
XG_model = XGBClassifier()
XG_model.fit(X_train, y_train)

y_predict = XG_model.predict(X_test)
y_predict_proba = XG_model.predict_proba(X_test)

xg_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
display('XGBoost model test set accuracy: {:.4f}'.format(xg_accuracy))
display('XGBoost model predicion distribution')
display(pd.Categorical(y_predict).describe())

display(pd.DataFrame(confusion_matrix(y_test, y_predict), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
display("auROC is: {}".format(roc_auc * 100))
```


    'XGBoost model test set accuracy: 0.9209'



    'XGBoost model predicion distribution'



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>848</td>
      <td>0.848849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151</td>
      <td>0.151151</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Predicted Not Churn</th>
      <th>Predicted to Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Not Churn</th>
      <td>786</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Actual Churn</th>
      <td>62</td>
      <td>134</td>
    </tr>
  </tbody>
</table>
</div>



    'auROC is: 96.4469972297761'


___What you should see___:

* Test set accuracy improved a little bit to 93% from 91%
* auROC improved a little bit too. Now at 94.8 


```python
#ROC chart
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
```


![png](output_51_0.png)


### Let's look at feature importance


```python
pd.DataFrame({'features': FEATURES, 'importance': XG_model.feature_importances_}).sort_values('importance', ascending=False)

```




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
      <th>features</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>TotalDayMinutes</td>
      <td>0.192371</td>
    </tr>
    <tr>
      <th>47</th>
      <td>CustomerServiceCalls</td>
      <td>0.092869</td>
    </tr>
    <tr>
      <th>54</th>
      <td>AvgRevPerServicePerMonth</td>
      <td>0.082919</td>
    </tr>
    <tr>
      <th>41</th>
      <td>TotalEveMinutes</td>
      <td>0.081260</td>
    </tr>
    <tr>
      <th>45</th>
      <td>TotalIntlMinutes</td>
      <td>0.053068</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Latency_share</td>
      <td>0.051410</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Contract_Month-to-month</td>
      <td>0.051410</td>
    </tr>
    <tr>
      <th>37</th>
      <td>tenure</td>
      <td>0.049751</td>
    </tr>
    <tr>
      <th>35</th>
      <td>InternationalPlan</td>
      <td>0.048093</td>
    </tr>
    <tr>
      <th>43</th>
      <td>TotalNightMinutes</td>
      <td>0.046434</td>
    </tr>
    <tr>
      <th>36</th>
      <td>VoiceMailPlan</td>
      <td>0.038143</td>
    </tr>
    <tr>
      <th>49</th>
      <td>TotalHighBandwidthMinutes</td>
      <td>0.036484</td>
    </tr>
    <tr>
      <th>51</th>
      <td>TotalRevenue</td>
      <td>0.026534</td>
    </tr>
    <tr>
      <th>46</th>
      <td>TotalIntlCalls</td>
      <td>0.024876</td>
    </tr>
    <tr>
      <th>7</th>
      <td>InternetService_Fiber optic</td>
      <td>0.019900</td>
    </tr>
    <tr>
      <th>26</th>
      <td>StreamingMovies_Yes</td>
      <td>0.013267</td>
    </tr>
    <tr>
      <th>44</th>
      <td>TotalNightCalls</td>
      <td>0.011609</td>
    </tr>
    <tr>
      <th>42</th>
      <td>TotalEveCalls</td>
      <td>0.008292</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PaymentMethod_Electronic check</td>
      <td>0.008292</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NumbervMailMessages</td>
      <td>0.008292</td>
    </tr>
    <tr>
      <th>48</th>
      <td>TotalCall</td>
      <td>0.008292</td>
    </tr>
    <tr>
      <th>40</th>
      <td>TotalDayCalls</td>
      <td>0.008292</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Contract_Two year</td>
      <td>0.006633</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PaperlessBilling</td>
      <td>0.006633</td>
    </tr>
    <tr>
      <th>50</th>
      <td>TotalHighLatencyMinutes</td>
      <td>0.006633</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TechSupport_No</td>
      <td>0.004975</td>
    </tr>
    <tr>
      <th>9</th>
      <td>OnlineSecurity_No</td>
      <td>0.004975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SeniorCitizen</td>
      <td>0.003317</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Contract_One year</td>
      <td>0.003317</td>
    </tr>
    <tr>
      <th>12</th>
      <td>OnlineBackup_No</td>
      <td>0.001658</td>
    </tr>
    <tr>
      <th>53</th>
      <td>num_services</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MaritalStatus</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MultipleLines</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PhoneService</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>InternetService_DSL</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dependents</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PaymentMethod_Bank transfer (automatic)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PaymentMethod_Mailed check</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PaymentMethod_Credit card (automatic)</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>OnlineSecurity_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>OnlineSecurity_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>OnlineBackup_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>OnlineBackup_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DeviceProtection_No</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DeviceProtection_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DeviceProtection_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TechSupport_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TechSupport_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>StreamingTV_No</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>StreamingTV_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>StreamingTV_Yes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>StreamingMovies_No</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>StreamingMovies_No internet service</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>InternetService_No</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>gender</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



___What you should see:___

* In fact, we see newly engineered features rising to the top of the importance
 * AvgRevPerServicePerMonth and Latency_share are in top 5
* Oddly - number of services is seemingly not relevant...

## Let's do a little bit of hyper-parameter tuning - let's play with depth only


```python
XG_model = XGBClassifier(max_depth = 8)
XG_model.fit(X_train, y_train)

y_predict = XG_model.predict(X_test)
y_predict_proba = XG_model.predict_proba(X_test)
xg_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
display('XGBoost model test set accuracy: {:.4f}'.format(xg_accuracy))
display('XGBoost model predicion distribution')
display(pd.Categorical(y_predict).describe())

display(pd.DataFrame(confusion_matrix(y_test, y_predict), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
display("auROC is: {}".format(roc_auc * 100))
```


    'XGBoost model test set accuracy: 0.9309'



    'XGBoost model predicion distribution'



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
      <th>counts</th>
      <th>freqs</th>
    </tr>
    <tr>
      <th>categories</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>834</td>
      <td>0.834835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165</td>
      <td>0.165165</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Predicted Not Churn</th>
      <th>Predicted to Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Not Churn</th>
      <td>784</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Actual Churn</th>
      <td>50</td>
      <td>146</td>
    </tr>
  </tbody>
</table>
</div>



    'auROC is: 97.12366889470607'


__What you should see:___

*Looks like max-depth of 5 raises auROC just a little bit
