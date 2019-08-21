import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib as ta
import seaborn as sns
import yfinance

from pandas_datareader import data as web
from sklearn import mixture as mix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = web.get_data_yahoo('SPY', start = '2000-01-01', end = '2017-08-08')
df = df[['Open', 'High', 'Low', 'Close']]

n = 10
t = 0.8
split = int(t * len(df))

df['high'] = df['High'].shift(1)
df['Low'] = df['Low'].shift(1)
df['Close'] = df['Close'].shift(1)

df['RSI'] = ta.RSI(np.array(df['close']), timeperiod = n) 
df['SMA'] = df['close'].rolling(window = n).mean()
df['Corr'] = df['SMA'].rolling(window = n).corr(df['close'])
df['SAR'] = ta.SAR(np.array(df['high']), np.array(df['low']), 0.2,0.2)
df['ADX'] = ta.ADX(np.array(df['high']), np.array(df['low']), \ 
                    np.array(df['close']), timeperiod = n)
df['Return'] = np.log(df['Open'] / df['Open'].shift(1))

print(df.head())

df = df.dropna()

ss = StandardScaler()
unsup.fit(np.reshape(ss.fit_transform(df[:split]), (-1, df.shape[1])))
regime = unsup.predict(np.reshape(ss.fit_transform(df[split:]), \
                            (-1, df.shape[1])))

Regimes = pd.DataFrame(regime, columns = ['Regime'], index=df[split:].index)\
                        .join(df[split:], how='inner') \
                            .assign(market_cu_return = df[split:] \
                                .Return.cumsum())\
                                .reset_index(drop = False) \
                                .rename(columns = {'index':'Date'})

order = [0,1,2,3]
fig = sns.FacetGrid(data = Regimes, hue='Regime' , hue_order=order, aspect=2 , size=4)
fig.map(plt.scater, 'Date', 'market_cu_return', s = 4).add_legend()
plt.show()

for i in order:
    print('MEan for regime %i:' %i, unsup.means_[i][0])
    print('Co-Varience for regime %i' %i, (unsupp.covariances_[i]))

ss1 = StandardScaler()
columns = Regimes.Columns.drop(['Regime', 'Date'])
Regimes[columns] = ss1.fit_transform(Regimes[columns])
Regimes['Signals'] = 0
Regimes.loc[Regimes['Returns'] >0, 'Signal'] = 1
Regimes.loc[Regimes['Returns'] <0, 'Signal'] = -1


cls = SVC(C=1.0, kernel='rbf', cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', max_iter=-1,
            probability=False, random_state=None,  shrinking=True, tol=1e-3, verbose=False  )


splits = int(0.8 * len(Regimes))

X = Regimes.drop(['Signal', 'Return', 'market_cu_return', 'Date'], axis = 1)
y = Regimes['Signal']

cls.fit(X[:splits], y[:split2])

p_data = len(X) - split2

df['Pred_Signal'] = 0
df.iloc[-p_data:, df.columns.get_loc('Pred_Signal')] = cls.predict(X[split2:])

print(df['Pred_Signal'][-p_data:])

df['str_ret'] = df['Pred_Signal'] * df['Return'].shift(-1)

df['strategy_cu_return'] = 0
df['market_cu_return'] = 0
df.iloc[-p_data:, df.columns.get_loc('strategy_cu_return')] = \
            np.nancumsum(df['str_ret'][-p_data:])
Sharpe = (df['strategy_cu_return'][-1] - df['market_cu_return'][-1])\
            /np.nanstd(df['strategy_cu_return'][-p_data:])

plt.plot(df['strategy_cu_return'][-p_data:], color = 'g', label = 'Strategy Returns')
plt.plot(df['market_cu_return'][-p_data:], color = 'r', label = 'Market Returns')
plt.figtext(0.14, 0.9, s = 'Sharpe Ratio: %.2f'%Sharpe)
plt.legend(loc = 'best')
plt.show()
