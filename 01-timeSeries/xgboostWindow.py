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
from math import copysign

import pickle

from statsmodels.graphics.tsaplots import plot_acf

from collections import Counter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay


# %%

timeSeries=(pd.read_csv(r'C:\Users\yosty\Desktop\Desktop_Folder\14 - git\timeSeriesDOTS\timeSeriesDOTS\dots\00-data\clean\dotsTimeSeries.csv')
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


#%%
netStats=pd.read_csv(r'C:\Users\yosty\Desktop\Desktop_Folder\14 - git\timeSeriesDOTS\timeSeriesDOTS\dots\00-data\clean\DOTSnetStats.csv').drop(['Unnamed: 0', 'CONNECTIVITY', 'HAS_BRIDGE', 'TOTAL_NET_VALUE', 'PAGERANK_NUMPY'],axis=1)
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


#%%


def direction(y_test, y_pred):
    # if the test data is 0, no sign to get correct
    if y_test.iloc[0,0] == np.float32(0):
        return(0)

    # copies sign to 1
    testSign=copysign(1, y_test)
    predSign=copysign(1, y_pred)

    # if sign if correct and positive
    if testSign and predSign == 1:
        return(1)
    # if sign is correct and negative
    elif testSign and predSign == -1:
        return(-1)
    # not sure how to properly denote this
    # using - int 2 to say we predicted - but should be +
    elif testSign == 1 and predSign == -1:
        return(-2)
    elif testSign == -1 and predSign == 1:
        return(2)
    else:
        return(0)



# %%

####################################################### moving

# %%

# https://xgboost.readthedocs.io/en/latest/python/examples/index.html
# https://xgboost.readthedocs.io/en/stable/parameter.html
# https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn

# https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning

results={}

econs=pd.Series(col for col in tsPctChange.columns).unique()

# we want window to be 48
# + 6 rows to create lagged vars
# + 1 for test period

WINDOW_size=48
LAGS=12
FORECAST=1
window=WINDOW_size+LAGS+FORECAST

for tempSeries in tqdm(econs[0:1]):

    results[tempSeries]={}
    # create dataset

    # create dataset
    # network statistics
    X=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == tempSeries.split('-')[0] or col[0] == tempSeries.split('-')[1]]]
    X.columns=["-".join(col) for col in X.columns]
    X_temp=X

    # bilateral trade series
    y=tsPctChange[[tempSeries]]

    assert(len(y)==len(X))

    for j in range(0, len(y)-window):

        resultsKey=str(tempSeries+':'+str(j)+'-'+str(j+window))

        # create windowed data
        y_temp=y[[tempSeries]][j:j+window]
        X_temp=X[j:j+window]

        # if there is data for model
        if not X_temp.empty and not y.empty:

            # create lagged dependent var features
            for i in range(1,LAGS+1):
                X_temp=X_temp.merge(y_temp.shift(i), left_index=True, right_index=True)
                X_temp.rename({tempSeries: f'y-{i}'}, axis=1, inplace=True)

            X_temp=X_temp.iloc[5:]
            y_temp=y_temp.iloc[5:]

            results[tempSeries][resultsKey]={}
            results[tempSeries][resultsKey]['window'] = (j, j+window)
            results[tempSeries][resultsKey]['y_std']=y.std()
            results[tempSeries][resultsKey]['series']=tempSeries
            # just forecast next period
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=int(1), shuffle=False)
            results[tempSeries][resultsKey]['y_test_std']=y_test.std()

            #bestParams=hyperParameterTuning(X_train, y_train)

            bst = xgb.XGBRegressor(
                # https://xgboost.readthedocs.io/en/stable/parameter.html
                objective = 'reg:pseudohubererror',#'reg:squarederror',
                colsample_bytree = .05,
                learning_rate = .1,
                max_depth = 3,
                min_child_weight = 5,
                n_estimators = 500,
                subsample = .5,
                nthread=4)

            #results[tempSeries]['bestParams']=bestParams

            bst.fit(X_train, y_train)

            results[tempSeries][resultsKey]['model']=bst

            y_pred = pd.DataFrame(bst.predict(X_test))
            y_pred.index=y_test.index

            mse=mean_squared_error(y_test, y_pred)
            results[tempSeries][resultsKey]['mse']=mse

            results[tempSeries][resultsKey]['data']=[X_train, X_test, y_train, y_test, y_pred]
            results[tempSeries][resultsKey]['test_direction']=copysign(1, y_test.iloc[0,0])
            results[tempSeries][resultsKey]['pred_direction']=copysign(1, y_pred.iloc[0,0])

            importances=['weight', 'gain', 'cover']
            for importance in importances:
                results[tempSeries][resultsKey][importance]=(bst.get_booster().get_score(importance_type=importance))




#%%

import statistics
calculationsDict={}

for seriesKey in results.keys():
    mseList=[]
    testDirection=[]
    predDirection=[]
    for windowSeriesKey in results[seriesKey].keys():
        mseList.append(results[seriesKey][windowSeriesKey]['mse'])
        testDirection.append(results[seriesKey][windowSeriesKey]['test_direction'])
        predDirection.append(results[seriesKey][windowSeriesKey]['pred_direction'])
    calculationsDict[seriesKey]={}
    calculationsDict[seriesKey]['mseList']=mseList
    calculationsDict[seriesKey]['avgMSE']=statistics.mean(mseList)
    calculationsDict[seriesKey]['cm']=confusion_matrix(testDirection, predDirection)



#%%

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

for seriesKey in calculationsDict.keys():
    ConfusionMatrixDisplay(confusion_matrix=calculationsDict[seriesKey]['cm'], display_labels=[1.0,-1.0]).plot()


#%%

# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
# y_score = clf.decision_function(X_test)

# fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
# pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# roc_display.plot(ax=ax1)
# pr_display.plot(ax=ax2)
# plt.show()


#%%

good=[series for series in calculationsDict.keys() if calculationsDict[series]['avgMSE'] < .005]
good

#%%

import plotly.express as px
import plotly.graph_objects as go


for seriesKey in good:
    actual=[]
    pred=[]
    for windowSeriesKey in results[seriesKey].keys():
        modelData=results[seriesKey][windowSeriesKey]['data']
        actual.append(pd.DataFrame(modelData[3]))
        pred.append(pd.DataFrame(modelData[4]))

    actual=pd.concat(actual)
    pred=pd.concat(pred)

    plotData=pd.merge(actual, pred, left_index=True, right_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plotData.index, y=plotData.iloc[:,0],
                    mode='lines',
                    name='actual'))
    fig.add_trace(go.Scatter(x=plotData.index, y=plotData.iloc[:,1],
                    mode='markers',
                    name='pred'))
    fig.show()


#%%



