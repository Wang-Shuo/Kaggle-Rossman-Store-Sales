"""
this script performs parameter tuning process on the model ID 24086
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


num_boost_round = 5000

model_dict = {}

model_num = 0
# parameter tuning process       
for max_depth in range(8,11):
    for subsample in range(7,10):
        for colsample_bytree in range(5,7):

            params = {'objective': 'reg:linear',
                       'booster': 'gbtree',
                       'eta': 0.03,
                       'silent': 1,
                       'seed': 1301
                      }


            params['colsample_bytree'] = colsample_bytree * 0.1
            params['subsample'] = subsample * 0.1
            params['max_depth'] = max_depth

            print("train {} xgboost model".format(model_num))
            dtrain = xgb.DMatrix(X_train[features_used], y_train)
            dvalid = xgb.DMatrix(X_valid[features_used], y_valid)


            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
                             early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)


            print('performing validation')
            yhat = gbm.predict(xgb.DMatrix(X_valid[features_used]))
            error = rmspe(X_valid.Sales.values, np.expm1(yhat))
            print('RMSPE: {:.6f}'.format(error))

            params['valid_error'] = error
            model_dict[str(model_num)] = params

            print('Make predictions on the test set')
            dtest = xgb.DMatrix(test_df[features_used])
            test_probs = gbm.predict(dtest)

            # output
            result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
            result.to_csv('output/test_prediction/xgb_24086_{}.csv'.format(model_num), index=False)

            model_num += 1


models_df = pd.DataFrame(model_dict).T 
models_df.to_csv('output/xgb_24086_param_tuning.csv')
