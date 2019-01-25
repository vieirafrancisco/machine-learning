import pandas as pd
import quandl
import pickle

df = quandl.get("WIKI/GOOGL")
print(df.shape)

## save the classifier with pickle
with open('lin_reg.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('lin_reg.pickle', 'rb')
clf = pickle.load(pickle_in)