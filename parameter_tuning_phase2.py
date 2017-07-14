"""
this script tunes the following parameters based on the phase1 best parameters:
      eta: 0.03 --> 0.01
      num_boost_round:  5000--> 15000
      early_stopping_rounds: 100 --> 600

"""

import pandas as pd
import numpy as np
from preprocess import preprocess
from extract_features import extract_features
import xgboost as xgb


# load the data
print("Load the train, test and store data")
train = pd.read_csv("input/train.csv", parse_dates=[2])
test = pd.read_csv("input/test.csv", parse_dates=[3])
store = pd.read_csv("input/store.csv")

# preprocess the data
print("Preprocess the data")
preprocessed_df = preprocess(train, test, store)

# define a feature list to store all feature names
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


# merge the features from 5 models above in the model description
features_used = []
for model_index in [24086]:
	with open('output/features/{}.txt'.format('features_' + str(model_index)), 'r') as ft:
		fts = ft.readlines()[0].split("'")
		fts = [f for f in fts if len(f) > 2]
		features_used += fts

features_used = list(set(features_used))


num_boost_round = 15000


params = {'objective': 'reg:linear',
          'booster': 'gbtree',
          'eta': 0.01,
          'colsample_bytree': 0.5,
          'subsample': 0.7,
          'max_depth': 10,
          'silent': 1,
          'seed': 1301
         }


print("train xgboost model")
dtrain = xgb.DMatrix(X_train[features_used], y_train)
dvalid = xgb.DMatrix(X_valid[features_used], y_valid)


watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
                 early_stopping_rounds=600, feval=rmspe_xg, verbose_eval=True)


print('performing validation')
yhat = gbm.predict(xgb.DMatrix(X_valid[features_used]))
valid_result = pd.DataFrame({'Sales': np.expm1(yhat)})
valid_result.to_csv('output/valid_prediction/valid_24086_tuned.csv', index=False)
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print('Make predictions on the test set')
dtest = xgb.DMatrix(test_df[features_used])
test_probs = gbm.predict(dtest)

# output
result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
result.to_csv('output/test_prediction/xgb_24086_tuned.csv', index=False)


