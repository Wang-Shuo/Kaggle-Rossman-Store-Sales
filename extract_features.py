import pandas as pd 
import numpy as np 


def extract_features(features, data):

	"""
	this module deals with feature engineering.
	The parameter 'features' is a list that saves the feature name; 'data' represents the dataframe which will be extracted features from. 
	"""
	# Promo open time in months
	features.append('PromoOpenInMonth')
	data['PromoOpenInMonth'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
	data['PromoOpenInMonth'] = data.PromoOpenInMonth.apply(lambda x: x if x > 0 else 0)
	data.loc[data.Promo2SinceYear == 0, 'PromoOpenInMonth'] = 0

	# calculate time competition open time on months
	features.append('CompetitionOpenInMonth')
	data['CompetitionOpenInMonth'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (data.Month - data.CompetitionOpenSinceMonth)

	# indicate that sales on that day are in promo interval
	features.append('IsPromoMonth')
	month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
	data['monthStr'] = data.Month.map(month2str)
	data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
	data['IsPromoMonth'] = 0
	for interval in data.PromoInterval.unique():
		if interval != '':
			for month in interval.split(','):
				data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

	# calculate average sales per store, customers per store, customer median and sales per customers
	features.extend(['AvgSales', 'AvgCustomers', 'AvgSalesPerCustomer', 'medianCustomers'])

	#1 Get total sales, customers and open days per store
	train_df = data[data['Set'] == 1]
	totalSalesPerStore = train_df.groupby([train_df['Store']])['Sales'].sum()
	totalCustomersPerStore = train_df.groupby([train_df['Store']])['Customers'].sum()
	totalOpenStores = train_df.groupby([train_df['Store']])['Open'].count()
	medianCustomers = train_df.groupby([train_df['Store']])['Customers'].median()

	#2 compute averages
	AvgSales = totalSalesPerStore / totalOpenStores
	AvgCustomers = totalCustomersPerStore / totalOpenStores
	AvgSalesPerCustomer = AvgSales / AvgCustomers

	#3 merge the averages into data
	data = pd.merge(data, AvgSales.reset_index(name='AvgSales'), how='left', on=['Store'])
	data = pd.merge(data, AvgCustomers.reset_index(name='AvgCustomers'), how='left', on=['Store'])
	data = pd.merge(data, AvgSalesPerCustomer.reset_index(name='AvgSalesPerCustomer'), how='left', on=['Store'])
	data = pd.merge(data, medianCustomers.reset_index(name='medianCustomers'), how='left', on=['Store'])

	# calculate number of schoolholidays this week, last week and next week
	features.extend(['holidays_thisweek', 'holidays_lastweek', 'holidays_nextweek'])

	holidays_count = data.groupby(['Store','Year','WeekOfYear'])['SchoolHoliday'].sum().reset_index(name='holidays_thisweek')
	holidays_count['holidays_lastweek'] = 0
	holidays_count['holidays_nextweek'] = 0

	for store_id in holidays_count.Store.unique().tolist():
		store_lgt = len(holidays_count[holidays_count['Store'] == store_id])
		holidays_count.loc[1:store_lgt-1, 'holidays_lastweek'] = holidays_count.loc[0:store_lgt-2, 'holidays_thisweek'].values
		holidays_count.loc[0:store_lgt-2, 'holidays_nextweek'] = holidays_count.loc[1:store_lgt-1, 'holidays_thisweek'].values

	data = pd.merge(data, holidays_count, how='left', on=['Store', 'Year', 'WeekOfYear'])

	# calculate average and median sales and customers per store per day of week
	features.extend(['AvgSalesPerDow', 'medianSalesPerDow', 'AvgCustsPerDow', 'medianCustsPerDow'])

	AvgSalesPerDow = train_df.groupby(['Store', 'DayOfWeek'])['Sales'].mean()
	medianSalesPerDow = train_df.groupby(['Store', 'DayOfWeek'])['Sales'].median()
	AvgCustsPerDow = train_df.groupby(['Store', 'DayOfWeek'])['Customers'].mean()
	medianCustsPerDow = train_df.groupby(['Store', 'DayOfWeek'])['Customers'].median()

	# merge
	data = pd.merge(data, AvgSalesPerDow.reset_index(name='AvgSalesPerDow'), how='left', on=['Store', 'DayOfWeek'])
	data = pd.merge(data, AvgCustsPerDow.reset_index(name='AvgCustsPerDow'), how='left', on=['Store', 'DayOfWeek'])
	data = pd.merge(data, medianSalesPerDow.reset_index(name='medianSalesPerDow'), how='left', on=['Store', 'DayOfWeek'])
	data = pd.merge(data, medianCustsPerDow.reset_index(name='medianCustsPerDow'), how='left', on=['Store', 'DayOfWeek'])


	features.extend(['Store','CompetitionDistance','Promo','Promo2', \
					 'StoreType','Assortment','StateHoliday', 'SchoolHoliday', \
					 'Year','Month','WeekOfYear','DayOfWeek','DayOfMonth', \
						 'DayOfYear'])

	return data


	






