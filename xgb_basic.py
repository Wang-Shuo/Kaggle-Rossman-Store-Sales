"""
this script trains a basic xgboost model using all features

# model description: use all 28 features
#features used: ['PromoOpenInMonth', 'CompetitionOpenInMonth', 'IsPromoMonth', 'AvgSales', 
           'AvgCustomers', 'AvgSalesPerCustomer', 'medianCustomers', 'holidays_thisweek', 
            'holidays_lastweek', 'holidays_nextweek', 'AvgSalesPerDow', 'medianSalesPerDow', 
            'AvgCustsPerDow', 'medianCustsPerDow', 'Store', 'CompetitionDistance', 'Promo', 
           'Promo2', 'StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 'Year', 
            'Month', 'WeekOfYear', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']

# train rmspe: 0.076564
# validation RMSPE: 0.115569
"""


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from preprocess import preprocess
from extract_features import extract_features
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 


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


# create feature map
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feature in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feature))
    outfile.close()


params = {'objective': 'reg:linear',
          'booster': 'gbtree',
          'eta': 0.03,
          'max_depth': 10,
          'subsample': 0.9,
          'colsample_bytree': 0.5,
          'silent': 1,
          'seed': 1301
         }
num_boost_round = 5000

print("train a xgboost model")
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)


watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
               early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)


print('performing validation')
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))

valid_result = pd.DataFrame({'Sales': np.expm1(yhat)})
valid_result.to_csv('output/valid_prediction/valid_basic.csv', index=False)
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print('Make predictions on the test set')
dtest = xgb.DMatrix(test_df[features])
test_probs = gbm.predict(dtest)

# output
result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
result.to_csv('output/test_prediction/xgb_basic.csv', index=False)


# XGB feature importances
print('create feature map to get feature importance')
create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv('output/feature_importance_xgb.csv', index=False)

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('output/feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
