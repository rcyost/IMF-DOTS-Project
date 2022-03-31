# %% tags=["parameters"]
upstream = ['calculateNetworkStats', 'createTimeSeries'] # this means: execute raw.py, then clean.py
product = None


# %% [markdown]
#    In [previous work](https://rcyost.github.io/DOTS-network) I've calculated some basic network statistics on IMF Direction of Trade Statistics (DOTS) export data.
# 
#    I looked for relevant features using univariate linear regression [in this notebook](https://rcyost.github.io/network-feature-engineering-trade)
# 
#    In this notebook I'll use XGBoost.
# 
#   Table of Contents:
#    1. Load and clean data
#    2. For each trade series, XGBoost export series against the exporter's network statistics
#       - This could be re-run on importer's statistics
#       - Recalculate the network with edges as nodes: [example](https://youtu.be/p5LO97n3llg?t=235)
#    3. Sort by mean absolute error.
#    4. Collapse network statistics with PCA, repeat 2,3,4 on PCA series
# 


# %% [markdown]
#    ### 1. Load and clean data

# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


from numpy import absolute
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


from tqdm import tqdm

from math import ceil

import pickle

# %%

timeSeries=(pd.read_csv(upstream['createTimeSeries']['dotsTimeSeries'])
    .pivot_table(index='period', columns=['ReferenceArea', 'CounterpartReferenceArea'], values='value')
)
# timeSeries=(pd.read_csv('dotsTimeSeries.csv')
#     .pivot_table(index='period', columns=['ReferenceArea', 'CounterpartReferenceArea'], values='value')
# )


tsPctChange=np.log(timeSeries).pct_change().iloc[1:].dropna(axis=1)
tsPctChange.columns=['-'.join(col) for col in tsPctChange.columns]
tsPctChange[tsPctChange>1.5]=np.nan
tsPctChange[tsPctChange<-1.5]=np.nan
tsPctChange=tsPctChange.dropna(axis=1)
tsPctChange.index=pd.to_datetime(tsPctChange.index)
tsPctChange=tsPctChange[tsPctChange.index > '1985-01-01']

netStats=pd.read_csv(upstream['calculateNetworkStats']['DOTSnetStats']).drop(['Unnamed: 0', 'CONNECTIVITY', 'HAS_BRIDGE', 'TOTAL_NET_VALUE', 'PAGERANK_NUMPY'],axis=1)
# netStats=pd.read_csv('DOTSnetStats.csv').drop(['Unnamed: 0', 'CONNECTIVITY', 'HAS_BRIDGE', 'TOTAL_NET_VALUE', 'PAGERANK_NUMPY'],axis=1)
netStats.set_index(['index', 'PERIOD'], inplace=True)
# get to period index and econ, stats cols
netStatsWide=(netStats
.reset_index()
.melt(id_vars=['index', 'PERIOD'])
.pivot_table(index='PERIOD', columns=['index', 'variable'], values='value')
)
netStatsWide.index = pd.to_datetime(netStatsWide.index)
netStatsWidePctChange=netStatsWide.pct_change().iloc[1:].dropna(axis=1)
netStatsWidePctChange.index=pd.to_datetime(netStatsWidePctChange.index)
netStatsWidePctChange=netStatsWidePctChange[netStatsWidePctChange.index > '1985-01-01']


# %%

# lag the net stats to not leak information
netStatsWidePctChange=netStatsWidePctChange.shift(-1).iloc[:-1]
# take off a period of time series so sizes match
tsPctChange=tsPctChange.iloc[:-1]

# %%
netStats.corr()


# %%
netStatsWidePctChange.head()


# %%
netStatsWidePctChange.corr()


# %%
tsPctChange.head()

# %%
importers=pd.Series(col.split('-')[0] for col in tsPctChange.columns).unique()
exporters=pd.Series(col.split('-')[1] for col in tsPctChange.columns).unique()
allEcons=sorted(set(list(importers) + list(exporters)))
netStats=pd.Series(col[1] for col in netStatsWidePctChange.columns).nunique()

print('The upper-bound on number of tests:', len(allEcons)*netStats)

# %% [markdown]
#   ## 2. Loop and XGBoost

# %%

# https://www.kaggle.com/felipefiorini/xgboost-hyper-parameter-tuning
# https://www.kaggle.com/felipefiorini/xgboost-hyper-parameter-tuning/notebook

def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = xgb.XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 0)

    gsearch.fit(X_train,y_train)

    return(gsearch.best_params_)

# %%

# https://xgboost.readthedocs.io/en/latest/python/examples/index.html
# https://xgboost.readthedocs.io/en/stable/parameter.html
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn

# https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning

results={}

econs=pd.Series(col for col in tsPctChange.columns).unique()
tempSeries='Argentina-Brazil'
for tempSeries in tqdm(econs):
    # create dataset
    # network statistics
    X=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == tempSeries.split('-')[0] or col[0] == tempSeries.split('-')[1]]]
    X.columns=["-".join(col) for col in X.columns]
    X_temp=X

    # bilateral trade series
    y=tsPctChange[[tempSeries]]

    # if there is data for model
    if not X_temp.empty and not y.empty:
        results[tempSeries]={}
        results[tempSeries]['y_std']=y.std()
        results[tempSeries]['series']=tempSeries
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.1, shuffle=False)
        results[tempSeries]['y_test_std']=y_test.std()

        bestParams=hyperParameterTuning(X_train, y_train)

        bst = xgb.XGBRegressor(
            objective = 'reg:squarederror',
            colsample_bytree = bestParams['colsample_bytree'],
            learning_rate = bestParams['learning_rate'],
            max_depth = bestParams['max_depth'],
            min_child_weight = bestParams['min_child_weight'],
            n_estimators = bestParams['n_estimators'],
            subsample = bestParams['subsample'],
            nthread=4)

        results[tempSeries]['bestParams']=bestParams

        bst.fit(X_train, y_train)

        results[tempSeries]['model']=bst

        y_pred = bst.predict(X_test)

        mse=mean_squared_error(y_test, y_pred)
        results[tempSeries]['mse']=mse

        results[tempSeries]['data']=[X_train, X_test, y_train, y_test, y_pred]


        importances=['weight', 'gain', 'cover']
        for importance in importances:
            results[tempSeries][importance]=(bst.get_booster().get_score(importance_type=importance))

#%%

with open(str(product['resultsDict']), 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



# %%

params=['learning_rate',
        'max_depth',
        'min_child_weight',
        'subsample',
        'colsample_bytree',
        'n_estimators',
        'objective']
ncols=4

nrows = ceil(len(params) / ncols)

width = ncols * 5
length = nrows * 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))

for param, ax in zip(params, axes.flatten()):
    brParams=pd.DataFrame([results[key]['bestParams'][param] for key in list(results.keys())])
    brParams.columns=[param]
    counts=pd.DataFrame(brParams[param].value_counts())
    ax.barh(counts.index.astype('str'), counts.iloc[:,0])
    ax.set_title(param)

plt.tight_layout()


# %%

importances=['weight', 'gain', 'cover']

for importance in importances:
    plt.figure()

    df=pd.DataFrame([results[key][importance] for key in list(results.keys())])
    width = ncols * 5
    length = nrows * 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))
    fig.suptitle(importance)
    nrows = ceil(len(pd.Series(col.split('-')[1] for col in df.columns).unique())/ ncols)

    df=df.melt()
    df[['econ', 'netStat']] = df['variable'].str.split('-', 1, expand=True)
    df
    df=df[['netStat', 'value']].pivot(columns='netStat')
    for i, col in enumerate(df.columns):
        df[[col]].dropna().hist(ax=axes.flatten()[i])


plt.tight_layout()

# %%
pd.DataFrame([results[key]['mse'] for key in list(results.keys())]).hist(bins=100)

# %%

good=[results[key]['series'] for key in list(results.keys()) if results[key]['mse'] < .0002]
good

# %%

bad=[results[key]['series'] for key in list(results.keys()) if results[key]['mse'] > .05]
bad

# %%

def plotSeries(inputSeries):
    for i, series in enumerate(inputSeries):
        plt.figure()
        modelData=results[inputSeries[i]]['data']

        plt.title(results[inputSeries[i]]['series'])
        plt.plot(modelData[3].values, 'g')
        plt.plot(modelData[4], '*b')


# %%

plotSeries(good)


# %%

plotSeries(bad)


# %%

scatterDF=[]
for key in list(results.keys()):
    scatterDF.append(pd.DataFrame(
    {'mse':results[key]['mse'],
    'y_std':results[key]['y_std']}))

scatterDF=pd.concat(scatterDF)
scatterDF.plot.scatter(x='mse', y='y_std')
plt.title('train y std dev')


# %%

scatterDF=[]
for key in list(results.keys()):
    scatterDF.append(pd.DataFrame(
    {'mse':results[key]['mse'],
    'y_std':results[key]['y_test_std']}))

scatterDF=pd.concat(scatterDF)
scatterDF.plot.scatter(x='mse', y='y_std')
plt.title('test y std dev')


# %% [markdown]
#   It is easier to forecast out of sample on series with lower standard deviation

# %% [markdown]
#   ## 5. PCA on Network Statistics to Reduce Dimensionality

# %%

# https://xgboost.readthedocs.io/en/latest/python/examples/index.html
# https://xgboost.readthedocs.io/en/stable/parameter.html
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn


pcaResults={}

econs=pd.Series(col for col in tsPctChange.columns).unique()
for tempSeries in tqdm(econs):
    # create dataset
    # network statistics
    X=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == tempSeries.split('-')[0]]]
    X.columns=[col[1] for col in X.columns]
    X_temp=X

    # bilateral trade series
    y=tsPctChange[[tempSeries]]

    # if there is data for model
    if not X_temp.empty and not y.empty:
        pcaResults[tempSeries]={}
        pcaResults[tempSeries]['y_std']=y.std()
        pcaResults[tempSeries]['series']=tempSeries

        scaler = StandardScaler()
        scaledData = pd.DataFrame(scaler.fit_transform(X_temp))

        #####   PCA
        # create model
        n_components=4
        pcaModel = PCA(n_components=n_components)

        # fit model
        pcaModelFit = pcaModel.fit(scaledData)
        X_temp = pd.DataFrame(pcaModelFit.transform(scaledData), columns=[str(col) for col in range(n_components)])

        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.1, shuffle=False)
        pcaResults[tempSeries]['y_test_std']=y_test.std()

        bestParams=hyperParameterTuning(X_train, y_train)

        pcaResults[tempSeries]['bestParams']=bestParams

        bst = xgb.XGBRegressor(
            objective = 'reg:squarederror',
            colsample_bytree = bestParams['colsample_bytree'],
            learning_rate = bestParams['learning_rate'],
            max_depth = bestParams['max_depth'],
            min_child_weight = bestParams['min_child_weight'],
            n_estimators = bestParams['n_estimators'],
            subsample = bestParams['subsample'],
            nthread=4)

        bst.fit(X_train, y_train)

        pcaResults[tempSeries]['model']=bst

        y_pred = bst.predict(X_test)

        mse=mean_squared_error(y_test, y_pred)
        pcaResults[tempSeries]['mse']=mse

        pcaResults[tempSeries]['data']=[X_train, X_test, y_train, y_test, y_pred]


        importances=['weight', 'gain', 'cover']
        for importance in importances:
            pcaResults[tempSeries][importance]=(bst.get_booster().get_score(importance_type=importance))

#%%

with open(str(product['pcaResultsDict']), 'wb') as handle:
    pickle.dump(pcaResults, handle, protocol=pickle.HIGHEST_PROTOCOL)



# %%

params=['learning_rate',
        'max_depth',
        'min_child_weight',
        'subsample',
        'colsample_bytree',
        'n_estimators',
        'objective']
ncols=4

nrows = ceil(len(params) / ncols)

width = ncols * 5
length = nrows * 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))

for param, ax in zip(params, axes.flatten()):
    brParams=pd.DataFrame([pcaResults[key]['bestParams'][param] for key in list(pcaResults.keys())])
    brParams.columns=[param]
    counts=pd.DataFrame(brParams[param].value_counts())
    ax.barh(counts.index.astype('str'), counts.iloc[:,0])
    ax.set_title(param)

plt.tight_layout()



# %%
df=pd.DataFrame([pcaResults[key][importance] for key in list(pcaResults.keys())])
df

# %%

importances=['weight', 'gain', 'cover']

for importance in importances:
    plt.figure()

    df=pd.DataFrame([pcaResults[key][importance] for key in list(pcaResults.keys())])
    width = ncols * 5
    length = nrows * 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))
    fig.suptitle(importance)
    nrows = ceil(len(pd.Series(col for col in df.columns).unique())/ ncols)

    for i, col in enumerate(df.columns):
        df[[col]].dropna().hist(ax=axes.flatten()[i])


plt.tight_layout()


# %%

plt.figure()
# pre PCA
pd.DataFrame([results[key]['mse'] for key in list(results.keys())]).hist(bins=100)
# PCA
pd.DataFrame([pcaResults[key]['mse'] for key in list(pcaResults.keys())]).hist(bins=100, color="k")



# %%
resultsDF=pd.DataFrame([results[key]['mse'] for key in list(results.keys())],
[key for key in list(results.keys())])
resultsDF.columns=['prePCA']

pcaResultsDF=pd.DataFrame([pcaResults[key]['mse'] for key in list(pcaResults.keys())],
[key for key in list(pcaResults.keys())])
pcaResultsDF.columns=['PCA']

resultsDF=resultsDF.join(pcaResultsDF, how='outer')
resultsDF['diff'] = resultsDF['PCA'] - resultsDF['prePCA']
ax=resultsDF['diff'].hist(bins=100)
ax.set_title('Change of Error after PCA')


# %%
improved=resultsDF['diff'].nsmallest()
improved


# %%
worse=resultsDF['diff'].nlargest()
worse


# %%

def plotPCAseries(inputSeries):
    for i, series in enumerate(inputSeries):
        plt.figure()
        pcaModelData=pcaResults[inputSeries[i]]['data']
        modelData=results[inputSeries[i]]['data']

        plt.title(results[inputSeries[i]]['series'])
        plt.plot(pcaModelData[3].values, 'g')
        plt.plot(modelData[4], '*b')
        plt.plot(pcaModelData[4], '*r')




# %% [markdown]
#  Red is PCA model forecast, blue is prePCA model forecast, green is actual.

# %%
plotPCAseries(improved.index)


# %%
plotPCAseries(worse.index)


# %%

scatterDF=[]
for key in list(pcaResults.keys()):
    scatterDF.append(pd.DataFrame(
    {'mse':pcaResults[key]['mse'],
    'y_std':pcaResults[key]['y_std']}))

scatterDF=pd.concat(scatterDF)
scatterDF.plot.scatter(x='mse', y='y_std')
plt.title('train y std dev')




# %%

scatterDF=[]
for key in list(pcaResults.keys()):
    scatterDF.append(pd.DataFrame(
    {'mse':pcaResults[key]['mse'],
    'y_std':pcaResults[key]['y_test_std']}))

scatterDF=pd.concat(scatterDF)
scatterDF.plot.scatter(x='mse', y='y_std')
plt.title('test y std dev')




