"""
this script applies linear regression on each store data

"""

import pandas as pd 
import numpy as np 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from preprocess import preprocess
from extract_features import extract_features


# load the data 
print("Load the train, test and store data")
train = pd.read_csv("input/train.csv", parse_dates=[2])
test = pd.read_csv("input/test.csv", parse_dates=[3])
store = pd.read_csv("input/store.csv")

# preprocess the data
print("Preprocess the data")
preprocessed_df = preprocess(train, test, store)

# define a feature list to store feature names
features = []
# extract features from preprocessed data
print("Extract features")
features_df = extract_features(features, preprocessed_df)


# Evaluation calculation
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


# split train and test set
train_df = features_df[features_df['Set'] == 1]
test_df = features_df[features_df['Set'] == 0]

# use the last 6 weeks of the train set as validation set
timeDelta = test_df.Date.max() - test_df.Date.min()
maxDate = train_df.Date.max()
minDate = maxDate - timeDelta
# valid_indices is a list of boolean values which are true when date is within the last 6 weeks of train_df
valid_indices = train_df['Date'].apply(lambda x: (x >= minDate and x <= maxDate))
# train_indices is list of boolean values to get the train set
train_indices = valid_indices.apply(lambda x: (not x))

# split the train and valid set
X_train = train_df[train_indices]
X_valid = train_df[valid_indices]
y_train = train_df['LogSales'][train_indices]
y_valid = train_df['LogSales'][valid_indices]

X_valid['pred_sales'] = 0
test_df['pred_sales'] = 0

# features used
#sub_features = ['PromoOpenInMonth', 'holidays_thisweek','Promo','SchoolHoliday', 'Year','Month','DayOfWeek', 'StateHoliday', 'AvgSales', 'AvgSalesPerCustomer']  
#sub_features = ['PromoOpenInMonth', 'holidays_thisweek','Promo','SchoolHoliday', 'AvgSales', 'AvgSalesPerCustomer']  
#sub_features = ['PromoOpenInMonth', 'holidays_thisweek','Promo','SchoolHoliday']    
#sub_features = ['PromoOpenInMonth', 'holidays_thisweek','Promo','DayOfWeek']    
#sub_features = ['holidays_thisweek','Promo','DayOfWeek']  
sub_features = ['Promo','DayOfWeek']    # lr_02


# build the linear regression model for each store
for store_id in test_df['Store'].unique().tolist():

    lr = LinearRegression()
    tr_indices = (X_train['Store'] == store_id) 
    v_indices = (X_valid['Store'] == store_id)
    te_indices = (test_df['Store'] == store_id)
    lr.fit(X_train[tr_indices][sub_features], y_train[tr_indices])
    X_valid['pred_sales'][v_indices] = np.expm1(lr.predict(X_valid[v_indices][sub_features]))
    test_df['pred_sales'][te_indices] = np.expm1(lr.predict(test_df[te_indices][sub_features]))


# validation
valid_err = rmspe(X_valid.Sales.values, X_valid.pred_sales.values)
print("Validation RMSPE: {:.6f}".format(valid_err))

# make submission
result = pd.DataFrame({'Id': test['Id'], 'Sales': test_df.pred_sales.values})
result.to_csv('output/test_prediction/lr_02.csv', index=False)