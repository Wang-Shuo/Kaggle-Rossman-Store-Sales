"""
this script runs 100 models on random selections of the features

it proves that the model ID 24086 has the least validation error in the end.
"""


import pandas as pd
import numpy as np
from preprocess import preprocess
from extract_features import extract_features
import xgboost as xgb
import random

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

# save features list as a txt file
def save_features(features, i, k):
	fn = 'features_' + str(16+k) + '{0:0=3d}'.format(i) + '.txt'
	with open('output/features/{}'.format(fn), 'w') as outfile:
		outfile.write(str(features))


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


params = {'objective': 'reg:linear',
          'booster': 'gbtree',
          'eta': 0.03,
          'max_depth': 10,
          'subsample': 0.9,
          'colsample_bytree': 0.5,
          'silent': 1,
          'seed': 1301
         }
num_boost_round = 3000

# the number of feature selections performed 
num_of_models = 100
basic_features = ['CompetitionOpenInMonth','DayOfYear','DayOfMonth','PromoOpenInMonth',
				  'AvgSalesPerDow','WeekOfYear','AvgSales','medianSalesPerDow','AvgSalesPerCustomer',
				  'AvgCustsPerDow','CompetitionDistance','AvgCustomers','Store','medianCustsPerDow',
				  'medianCustomers','Month']
total_sample_features = ['holidays_thisweek','Year','DayOfWeek','Promo','IsPromoMonth','StoreType',
						 'SchoolHoliday','Assortment','Promo2','StateHoliday','holidays_lastweek', 
						 'holidays_nextweek']

# a dict to save 100 models' infos
model_dicts = dict()

for i in range(num_of_models):
	model_info = {}

	# pick k features from total sample features list and k starts from 4 and is added one every 20 iterations
	k = int(i/20) + 4
	sample_features = random.sample(total_sample_features, k)
	features_used = basic_features + sample_features
	model_info['features_used'] = features_used
	save_features(features_used, i, k)

	print("train No.{} xgboost model".format(i))
	dtrain = xgb.DMatrix(X_train[features_used], y_train)
	dvalid = xgb.DMatrix(X_valid[features_used], y_valid)


	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
               early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

	model_name = str(16+k) + '{0:0=3d}'.format(i)

	yhat = gbm.predict(xgb.DMatrix(X_valid[features_used]))
	valid_result = pd.DataFrame({'Sales': np.expm1(yhat)})
	valid_result.to_csv('output/valid_prediction/{}.csv'.format('valid_'+model_name), index=False)

	error = rmspe(X_valid.Sales.values, np.expm1(yhat))
	model_info['valid_error'] = error

	model_dicts[model_name] = model_info

	print('Make predictions on the test set')
	dtest = xgb.DMatrix(test_df[features_used])
	test_probs = gbm.predict(dtest)

	# output
	result = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs)})
	result.to_csv('output/test_prediction/{}.csv'.format('test_'+model_name), index=False)


models_df = pd.DataFrame(model_dicts).T 
models_df.to_csv('output/model_infos.csv')