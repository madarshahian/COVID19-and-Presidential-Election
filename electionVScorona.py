# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:03:09 2020

@author: madar
"""

import numpy as np
import pandas as pd
import theano as th
import pymc3 as pm
import matplotlib.pyplot as plt

#%%
Election_data = pd.read_csv('2016_US_County_Level_Presidential_Results.csv')
counties_population = pd.read_csv('counties_population.csv')
Election_data = Election_data.drop(columns=['Unnamed: 0'])
Election_data.drop_duplicates()
fib=[]
for i in counties_population['ID Geography'].values:
    fib.append(int(i[-5:]))
counties_population['ID Geography'] = fib
#%%
from datetime import date,timedelta
last_updated_date = date.today()
delta = timedelta(days=1)
import requests
base_url =  'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'
url = base_url+last_updated_date.strftime("%m-%d-%Y")+'.csv'
while requests.get(url).status_code==404:
    last_updated_date-=delta
    url = base_url+last_updated_date.strftime("%m-%d-%Y")+'.csv'
print("last updated file found in %s"%last_updated_date)
    
df = pd.read_csv(url,index_col=0,parse_dates=[0])
df_usa_raw = df[df["Country_Region"]=="US"].dropna()
df_usa_raw_FIBS = df_usa_raw.index.values
for i in range(len(df_usa_raw_FIBS)):
    if not np.isnan(df_usa_raw_FIBS[i]):
        df_usa_raw_FIBS[i] = int(df_usa_raw_FIBS[i])
df_usa_raw['FIBS'] = df_usa_raw_FIBS
merged_data_ = Election_data.merge(df_usa_raw, how='inner', left_on='combined_fips', right_on='FIBS')
merged_data = merged_data_.merge(counties_population, how='inner', left_on='combined_fips', right_on='ID Geography')
#%%

data_final = merged_data[['votes_dem', 'votes_gop', 'Population (2018)','state_abbr', 'county_name','Confirmed', 'Deaths','Active']]
data_final['diff'] = (data_final['votes_dem']-data_final['votes_gop']).values
data_final['sign'] = data_final['diff'].apply(func = lambda x : 0 if x>0 else 1)
data_final['state_code']=pd.factorize(data_final['state_abbr'] )[0]

#%%
data_final = data_final.dropna()
counter = 0
for i in range(51):
    w = data_final[data_final['state_code']==i]['sign']
    if sum(w) == len(w) or sum(w) == 0:
        print(f"all counties are the same for {data_final[data_final['state_code']==i]['state_abbr'].values[0]}")
        data_final=data_final.drop(index=data_final[data_final['state_code']==i].index)
        counter+=1
#%%
        
data_final['state_code']=pd.factorize(data_final['state_abbr'] )[0]

Y_c = th.shared(data_final['Confirmed'].values)
Y_d = th.shared(data_final['Deaths'].values)
indc = th.shared(data_final['sign'].values)
state = th.shared(data_final['state_code'].values)
Y_total = th.shared(data_final['Population (2018)'].values)
#%%
with pm.Model() as model_US:    
    theta  =  pm.Beta('theta', alpha=1, beta=3, shape=2)
    p = pm.Binomial('obs', p=theta[indc], observed=Y_c, n=Y_total)
    trace_1 = pm.sample(1000, tune=4000,cores=1,chains=2,
                   nuts_kwargs={"target_accept":0.96,
                                "max_treedepth": 12})
#%%
import arviz as az
az.plot_trace(trace_1)
plt.figure()
plt.hist(100*trace_1['theta'][:,0],50,alpha=.8,color='b')#dem
plt.hist(100*trace_1['theta'][:,1],50,alpha=.8,color='r')#gop
#%%
with pm.Model() as model_state_wise:    
    theta  =  pm.Beta('theta', alpha=1, beta=3, shape=(51-counter,2))
    p = pm.Binomial('obs', p=theta[state,indc], observed=Y_c, n=Y_total)
    trace = pm.sample(1000, tune=4000,cores=1,chains=2,
                   nuts_kwargs={"target_accept":0.96,
                                "max_treedepth": 12})

#%%
for i in range(51-counter):
    plt.figure()
    plt.hist(100*trace['theta'][:,i,0],50,alpha=.8,color='b')#dem
    plt.hist(100*trace['theta'][:,i,1],50,alpha=.8,color='r')#gop
    plt.title(pd.factorize(data_final['state_abbr'] )[1][i])


