
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
df = None

```

__You should see:__

* 3333 rows of data across all 35 columns (different variables)
* lots of rows have text data that needs to be converted into numerical data, ideally one-hot encoded


## Explore the data: 

* Check  for nulls
* Check for outliers on data that is numerical

### Check for nulls and drop if any


```python
# #First replace all empty strings with NAN


# Check for any null values


display(df.shape)
df = df.dropna()
display(df.shape)


```

__What you should see__:

* We have only 5 null values
 * Think OK to drop
 * In real world, you'd want to document this AND maybe even try to impute the missing number (since only missing Total Reveneu)


### Check for outliers in numerical data 

__What you should see__:
* Nothing too suspicious, think can assume data is clean
* Revenue is not a float or int - we should fix that


```python
df['TotalRevenue']= pd.to_numeric(df['TotalRevenue'], errors='coerce').fillna(0, downcast='infer')
```

## Mechanical Feature Engineering

Set up the mapping for mechanical feature engineering step. Once you've completed this, take a look at how this is handled in the solution branch--there's a much quicker, cleaner way to do this! 

(Hint--think about how you could use list comprehensions for this!)


```python


# We use capitals for constant variables
TARGET = 'Churn'
ID =  'customerID'
FEATURES = df.columns.tolist()

# Remove customer id - it can't possibly have any predictive power
FEATURES.remove(ID)
#remove target from list of features
FEATURES.remove(TARGET)


```

## Split data


```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = None

# Lets verify that our splitting has the same distribution
display(Markdown("__Split in training set__"))
display(pd.Categorical(y_train[TARGET]).describe())
display(Markdown("__Split in test set__"))
display(pd.Categorical(y_test[TARGET]).describe())
```

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
linear_model = None
linear_model.fit(X_train, y_train)

display(Markdown('###### Performance on training set'))

# Get predictions for training set
y_lr_train = None

# Get predictions for testing set
y_lr_test = None

# Get test set probabilities
y_lr_test_proba = None


# See how well we did - get accuracy for training set
lr_accuracy = None
display('Linear model train set accuracy: {:.2f}%'.format(lr_accuracy*100))
display(Markdown('Linear model predicion distribution:'))
display(pd.Categorical(y_lr_test).describe())

# Get accuracy for testing set
lr_accuracy = None
display('Linear model test set accuracy: {:.2f}%'.format(lr_accuracy*100))
display(Markdown('Linear model predicion distribution:'))
display(pd.Categorical(y_lr_test).describe())

# Display confusion matrix
display(pd.DataFrame(confusion_matrix(y_test, y_lr_test), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))

# Calculate AUROC
false_positive_rate, true_positive_rate, thresholds = None
roc_auc = None
display(Markdown('\n\nauROC is: {}'.format(roc_auc * 100)))
```


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
intercept = None

# Get coefficients (same order as features fed into Logistic Regression) and build table
named_coefs = None

# put it in a nice dataframe and sort from most impact to least, sicne raw print-out is ugly
display("Intercept is {}".format(intercept[0]))
display(named_coefs)
```

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
# Fit and train the model
XG_model = None

# Get test predictions
y_predict = None

# Get test prediction probabilities 
y_predict_proba = None

# Calculate accuracy
xg_accuracy = None
display('XGBoost model test set accuracy: {:.4f}'.format(xg_accuracy))
display('XGBoost model predicion distribution')
display(pd.Categorical(y_predict).describe())

# Display confusion matrix as a dataframe
display(pd.DataFrame(confusion_matrix(y_test, y_predict), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))

# Calculate AUROC
false_positive_rate, true_positive_rate, thresholds = None
roc_auc = None
display("auROC is: {}".format(roc_auc * 100))
```

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

### Let's understand what's going on with features

Create a dataframe of features and feature importances on the line below (check the solution branch if you're not sure about the syntax!)

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
df_latency['TotalHighBandwidthMinutes'] = None
df_latency['TotalHighLatencyMinutes'] = None
df_latency['Latency_share'] = None
df_latency['Latency_share'] = None

# Now, plot a histogram on the line below

latency_bins = [df_latency['Latency_share'].min(), .1, .2, .3, .4, df_latency['Latency_share'].max()]
df_latency['LatencyShareBin'] = pd.cut(df_latency['Latency_share'], latency_bins,include_lowest=True).astype(str)

display(pd.crosstab(df_latency['LatencyShareBin'],df_latency['Churn'],normalize='index',margins=True))

       
```

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



```

___What you should see:___

* Looks like folks with multiple services are in fact more likely to churn out, but slightly (crosstab)
* When we plot average dollars paid per service per month we see that regardless of number of services, those with higher price paid are in fact somewhat more likely to churn-out.
* That signal is not very strong, but we can hope that ML algo will be able to make it useful

## Add Features into training/test data set


```python
df_fe = df.copy()



FEATURES = None

# Set string containing column name that will be our target
TARGET = None


```

### Split the data


```python
display(Markdown("__Orginal distribution__"))
display(pd.Categorical(df_transformed_target[TARGET]).describe())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = None

# Lets verify that our splitting has the same distribution
display(Markdown("__Split in training set__"))
display(pd.Categorical(y_train).describe())
display(Markdown("__Split in test set__"))
display(pd.Categorical(y_test).describe())
```

___What you should see___:

* Latency share and AvgRevServicePerMonth are both part of feature set
* We split training and test sets


## Re-run XGBoost


```python
# Instantiate and fit a model
XG_model = None 


# Get test predictions
y_predict = None

# Get test prediction probabilities
y_predict_proba = None


# Get accuracy score
xg_accuracy = None
display('XGBoost model test set accuracy: {:.4f}'.format(xg_accuracy))
display('XGBoost model predicion distribution')
display(pd.Categorical(y_predict).describe())

# Display confusion matrix
display(pd.DataFrame(confusion_matrix(y_test, y_predict), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))

# Calculate AUROC
false_positive_rate, true_positive_rate, thresholds = None
roc_auc = None
display("auROC is: {}".format(roc_auc * 100))
```

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

### Let's look at feature importance


```python
pd.DataFrame({'features': FEATURES, 'importance': XG_model.feature_importances_}).sort_values('importance', ascending=False)

```

___What you should see:___

* In fact, we see newly engineered features rising to the top of the importance
 * AvgRevPerServicePerMonth and Latency_share are in top 5
* Oddly - number of services is seemingly not relevant...

## Let's do a little bit of hyper-parameter tuning - let's play with depth only


```python
# Try various models with different max_depth 
XG_model = None
XG_model.fit(X_train, y_train)


# Get predictions for test set
y_predict = None

# Get prediction probabilities for test set
y_predict_proba = None

# Calculate accuracy score
xg_accuracy = None
display('XGBoost model test set accuracy: {:.4f}'.format(xg_accuracy))
display('XGBoost model predicion distribution')
display(pd.Categorical(y_predict).describe())

display(pd.DataFrame(confusion_matrix(y_test, y_predict), 
             columns=['Predicted Not Churn', 'Predicted to Churn'],
             index=['Actual Not Churn', 'Actual Churn']))



# Now, calculate and display AUROC
false_positive_rate, true_positive_rate, thresholds = None
roc_auc = None
display("auROC is: {}".format(roc_auc * 100))
```

__What you should see:__

*Looks like max-depth of 5 raises auROC just a little bit
