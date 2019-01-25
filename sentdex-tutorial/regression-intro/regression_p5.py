import pandas as pd
import numpy as np
import quandl
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'PCT_change']]

forecast_col = 'Adj. Close'

#fill all na data to -99999
df.fillna(-99999, inplace=True)

#10% of the days prediction
forecast_out = int(math.ceil(0.01*len(df)))

#put in each row the forecast_col value up forecast_out times
df['label'] = df[forecast_col].shift(-forecast_out)

#features
X = df.drop(['label'], axis=1).values
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

original_df = df
df.dropna(inplace=True)

#label
y = df['label'].values

X_train, X_test, y_train, y_test = tts(X,y, test_size=0.2)

#classifier
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = original_df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # one day in timestamp
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    original_df.loc[next_date] = [np.nan for _ in range(len(original_df.columns)-1)] + [i]

original_df['Adj. Close'].plot()
original_df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
