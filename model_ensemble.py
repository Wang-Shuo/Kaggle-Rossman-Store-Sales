"""
blends the model 24086_tuned, 20003, 21038 and xgb_basic  
"""
import pandas as pd 

test = pd.read_csv("input/test.csv", parse_dates=[3])

xgb_basic = pd.read_csv('output/test_prediction/xgb_basic.csv')
xgb_24086_tuned = pd.read_csv('output/test_prediction/xgb_24086_tuned.csv')

test_20003 = pd.read_csv('output/test_prediction/test_20003.csv')
test_21038 = pd.read_csv('output/test_prediction/test_21038.csv')


blending = (xgb_basic.Sales + xgb_24086_tuned.Sales + test_20003.Sales + test_21038.Sales) * 0.995 / 4

# output
result = pd.DataFrame({'Id': test['Id'], 'Sales': blending})
result.to_csv('output/test_prediction/ensemble_b_24086t_20003_21038.csv', index=False)
