# -*- coding: utf-8 -*-
from __future__ import absolute_import

''' Generating Log Normal Regression curve for Bitcoin price projection. '''
__author__ = 'PyPatel'
__version__ = '2.0'

from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as LR

def noDays(day):
    delta = day - date(2009,1,3)
    return int(delta.days)

# Get data
data = pd.read_csv('./market-price.csv')
data.columns = ['Date', 'Price']
data['Date'] =pd.to_datetime(data.Date).dt.date

# Add column for No. Days since Bitcoin inception
data['Days'] = [noDays(i) for i in data['Date'].values]

# Remove 0 price days, since there is no data available for these days
data.drop(data[data['Price'] == 0].index, inplace = True)

# Take Logarithm based 10 of Days and Price, these are the variable to be optimise 
log_days = np.array(np.log10(data['Days'].values)).reshape(-1,1)
log_price = np.array(np.log10(data['Price'].values)).reshape(-1,1)

# Train Linear Regression model
reg = LR().fit(log_days, log_price)
print('Fitness Score of the model: ', reg.score(log_days, log_price))
print('Coefficient of the line: ', reg.coef_[0])
print('Intercept of the line: ', reg.intercept_) 

# Predictions
print('\n----------------------------------------')
print("Today's BTC Price based on regression: US ${}" .format(float(10**(reg.predict(np.log10(noDays(date.today())))))))
print('----------------------------------------')

# Visualization
plt.plot(log_price, label='BTC Price')
plt.plot(reg.coef_[0]*log_days + reg.intercept_ , label='Regression Line')
plt.legend()
plt.title('Lognormal regression for BTC daily price data')
plt.xlabel('No. of Days since BTC inception')
plt.ylabel(r"$\log_{10}(Price)$")
plt.show()
