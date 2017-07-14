"""
this script generates benchmark model 

"""

import pandas as pd 
import numpy as np 

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

test.loc[test['Open'].isnull(), 'Open'] = 1

cols = ['Store', 'DayOfWeek', 'Promo']

median_df = train.groupby(cols)['Sales'].median().reset_index()

test_pred = pd.merge(test, median_df, on=cols, how='left')
test_pred.loc[test_pred.Open == 0, 'Sales'] = 0

test_pred[['Id', 'Sales']].to_csv('output/test_prediction/benchmark.csv', index=False)
