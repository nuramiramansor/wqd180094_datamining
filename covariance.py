# In[96]:
import pandas as pd
from pandas import Series
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn import preprocessing


df = pd.read_csv('the_star_data.csv')
df.head(3)


# In[47]:
df.drop('ID', axis=1, inplace=True)
df.drop_duplicates(subset=['Datetime', 'CompanySymbol'], inplace=True)
df = df.set_index('Datetime')
df.sort_index(inplace=True)

df.head(3)


# In[51]:
def getData(name):
    filter = df['CompanySymbol']==name
    new_data = df[filter]
    return new_data

print('Created function.')


# In[64]:
maybank = getData('MAYBANK')
maybank = maybank[['LastPrice']]
maybank.head(10)


# In[65]:
maybank.plot()


# In[66]:
cimb = getData('CIMB')
cimb = cimb[['LastPrice']]
cimb.plot()


# In[71]:
result = maybank.merge(cimb, on = 'Datetime',how='outer')
result.columns = ['MAYBANK','CIMB']
result.head()


# In[73]:
from sklearn import preprocessing


# In[77]:
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(result)
scaled_df = pd.DataFrame(scaled)
scaled_df.columns = ['MAYBANK','CIMB']
scaled_df.head()


# In[78]:
scaled_df.cov()


# In[91]:
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
dataset = scaler.fit_transform(maybank.LastPrice)

n_paa_segments = 10
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))

plt.figure()
plt.subplot(2, 2, 1)  # First, raw time series
plt.plot(dataset[0].ravel(), "b-")
plt.title("Raw time series")


# In[92]:
plt.subplot(2, 2, 2)  # Second, PAA
plt.plot(dataset[0].ravel(), "b-", alpha=0.4)
plt.plot(paa_dataset_inv[0].ravel(), "b-")
plt.title("PAA")


# In[94]:
# SAX transform
n_sax_symbols = 8
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
sax_dataset_inv = sax.inverse_transform(sax.fit_transform(dataset))

plt.subplot(2, 2, 3)  # Then SAX
plt.plot(dataset[0].ravel(), "b-", alpha=0.4)
plt.plot(sax_dataset_inv[0].ravel(), "b-")
plt.title("SAX, %d symbols" % n_sax_symbols)


# In[95]:
# 1d-SAX transform
n_sax_symbols_avg = 8
n_sax_symbols_slope = 8
one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                alphabet_size_slope=n_sax_symbols_slope)
one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(dataset))

plt.subplot(2, 2, 4)  # Finally, 1d-SAX
plt.plot(dataset[0].ravel(), "b-", alpha=0.4)
plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
plt.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                          n_sax_symbols_avg,
                                          n_sax_symbols_slope))

plt.tight_layout()
plt.show()