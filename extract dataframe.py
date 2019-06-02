"""
This code extracts a time-series (Pandas dataframe) from the_star_data.csv. The resulting dataframe will be stored
in `all_df` variable. It lists the last prices for 1200+ companies sorted by the update time
"""


# ## Importing Libraries

# In[120]:
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings('ignore')


# ## Reading the Data

# In[2]:
df = pd.read_csv('the_star_data.csv')
df.drop('ID', axis=1, inplace=True)


# ## Taking a Look at the Data

# In[3]:
print(df.head(10))


# ## Dropping Duplicates

# In[4]:
df.drop_duplicates(subset=['Datetime', 'CompanySymbol'], inplace=True)


# ## Grouping Data by Company

# In[ ]:
grouped = df.groupby('CompanySymbol')


# ## Getting the Number of Unique Update-Datetimes for Companies
# In[127]:
sizes = []
for name, group in grouped:
    sizes.append(group.shape[0])
    
plt.figure(figsize=(14,8))
plt.hist(sizes, bins=200);


# ## Getting the Companies with 92+ Unique Update-Datetimes and Their Common Update-Datetimes

# In[128]:
datetimes = []
companies = []
for name, group in grouped:
    if group.shape[0] > 92:
        datetimes.append(group.Datetime.values)
        companies.append(name)
print(len(datetimes), 'companies with 92+ unique update-datetimes')
datetimes = [set(l) for l in datetimes]
intersection = set.intersection(*datetimes)
print(len(intersection), 'datetime common to all these companies')


# In[ ]:
all_df = pd.DataFrame({})
i = 0
for name, group in grouped:
    if name in companies:
        group = group[group.Datetime.isin(intersection)]
        i += 1
        print(i, end=' ')
        group['Datetime'] = group.Datetime.str.replace(r' - (\d):', r' - 0\1:')
        group['Datetime'] = pd.to_datetime(group.Datetime, format='%d %b %Y - %I:%M %p')
        group.index = group.Datetime
        group.drop(['CompanySymbol', 'OpenPrice', 'HighPrice', 'LowPrice', 'Datetime'], axis=1, inplace=True)
        col_names = all_df.columns.tolist()
        col_names.append(name)
        if all_df.shape == (0, 0):
            all_df = group
        else:
            all_df = all_df.join(group, how='outer')
        all_df.columns = col_names


# In[51]:
print('Resulting dataframe shape:', all_df.shape)
print('Resulting dataframe head:')
print(all_df.head())
print(all_df.isna().sum().sum(), 'null values')


# In[52]:
all_df.sort_index(inplace=True)


# In[54]:
plt.plot(all_df.DIGI);