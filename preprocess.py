
import pandas as pd 
import numpy as np 


def preprocess(train, test, store):

	"""
	preprocess the original train, test and store dataset including concat, date seperation, datatype transformation and dealing with NaN values 
	"""

	# add a flag variable to distinguish train and test dataset
	train['Set'] = 1
	test['Set'] = 0
	# combine the train and test set 
	train_test = pd.concat([train, test])


	# exclude zero sales in the training set
	train_test = train_test.loc[~((train_test['Set'] == 1) & (train_test['Sales'] == 0))]

	# convert number 0 in column 'StateHoliday' to string '0'
	train_test.loc[train_test['StateHoliday'] == 0, 'StateHoliday'] = '0'

	# seperate out the elements of the date column for train and test set
	train_test["Year"] = train_test["Date"].dt.year
	train_test["Month"] = train_test["Date"].dt.month
	train_test["DayOfMonth"] = train_test["Date"].dt.day
	train_test["WeekOfYear"] = train_test["Date"].dt.weekofyear
	train_test["DayOfYear"] = train_test["Date"].dt.dayofyear


	# Fill out NaN values with 1 in column 'Open' for test set
	train_test.loc[train_test['Open'].isnull(), 'Open'] = 1

	# Label encoding using panda category datatype. Label encoded Assortment and StoreType
	train_test['StateHoliday'] = train_test['StateHoliday'].astype('category').cat.codes
	train_test['DayOfWeek'] = train_test['DayOfWeek'].astype('category').cat.codes
	#train_test = pd.get_dummies(train_test, columns=['DayOfWeek', 'StateHoliday'])

	# Log Standardization ==> Better for RMSPE
	train_test.loc[train_test['Set'] == 1, 'LogSales'] = np.log1p(train_test.loc[train_test['Set'] == 1]['Sales'])


	# Preprocess the store data
	# Fill NaN values in store_df for "CompetitionDistance" = 0
	store["CompetitionDistance"][store["CompetitionDistance"].isnull()] = 0

	# Fill NaN values in store dataframe for "CompetitionSince[X]" with 1900-01
	store["CompetitionOpenSinceYear"][(store["CompetitionDistance"] != 0) & (store["CompetitionOpenSinceYear"].isnull())] = 1900
	store["CompetitionOpenSinceMonth"][(store["CompetitionDistance"] != 0) & (store["CompetitionOpenSinceMonth"].isnull())] = 1

	# Label encoded Assortment and StoreType
	store['Assortment'] = store['Assortment'].astype('category').cat.codes
	store['StoreType'] = store['StoreType'].astype('category').cat.codes
	#store = pd.get_dummies(store, columns=['Assortment', 'StoreType'])

	# merge train_test dataframe and store dataframe on column 'store'
	merged_df = pd.merge(train_test, store, how='left', on='Store')

	# Fill NaN values with 0
	merged_df.fillna(0, inplace=True)

	return merged_df


