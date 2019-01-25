import pandas as pd
import quandl
import math

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
df.dropna(inplace=True)

print(df.head())
