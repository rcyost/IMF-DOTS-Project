# %% tags=["parameters"]
upstream = ['calculateNetworkStats', 'createTimeSeries'] # this means: execute raw.py, then clean.py
product = None


# %% [markdown]
# -

# %% [markdown]
# -

# %% [markdown]
#  In [previous work](https://rcyost.github.io/DOTS-network) I've calculated some basic network statistics on IMF Direction of Trade Statistics (DOTS) export data.
# 
#  In this notebook I'll see if this data has any relationship with percent change bilateral export series. TLDR: currently no linear relationships
# 
# Table of Contents:
#  1. Load and clean data
#  2. For each trade series, univariate linear regress bilateral export series against the exporter's network statistics  
#     - This could be re-run on importer's statistics
#     - Recalculate the network with edges as nodes: [example](https://youtu.be/p5LO97n3llg?t=235)
#  3. Sortby pValue, r^2, and aic, check with plots
#  4. Collapse network statistics with PCA, repeat 2,3,4 on PCA series
# 
#  improvements / future work:
#  - dimension reduction
#      - elastic net selection
#      - PCA, DBSCAN,
#  - create more features
#      - rolling window
#      - expanding window
#      - lags
#      - use auto feature generation tools:
#          - https://www.featuretools.com/
#          - http://isadoranun.github.io/tsfeat/
#          - http://cesium-ml.org/
#          - https://tsfresh.readthedocs.io/en/latest/text/introduction.html
#          - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
#  - validate feature importance
#      - check autocorrelations
#      - check multicollinearity
#      - Random Forest importance
# 
#  - use univariate non-linear models
#  - use multivariate models
#  - use time series specific models
#  - use ml models
# 
# 
# 

# %% [markdown]
#  ### 1. Load and clean data

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# %%

timeSeries=(pd.read_csv(upstream['createTimeSeries']['dotsTimeSeries'])
    .pivot_table(index='period', columns=['ReferenceArea', 'CounterpartReferenceArea'], values='value')
)


tsPctChange=np.log(timeSeries).pct_change().iloc[1:].dropna(axis=1)
tsPctChange.columns=['-'.join(col) for col in tsPctChange.columns]
tsPctChange[tsPctChange>1.5]=np.nan
tsPctChange[tsPctChange<-1.5]=np.nan
tsPctChange=tsPctChange.dropna(axis=1)
tsPctChange.index=pd.to_datetime(tsPctChange.index)
tsPctChange=tsPctChange[tsPctChange.index > '1985-01-01']


netStats=pd.read_csv(upstream['calculateNetworkStats']['DOTSnetStats']).drop(['Unnamed: 0', 'CONNECTIVITY', 'HAS_BRIDGE', 'TOTAL_NET_VALUE', 'PAGERANK_NUMPY'],axis=1)
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
# ## 2. Loop and Linear Regress

# %%

econs=pd.Series(col for col in tsPctChange.columns).unique()
regResults=[]
for tempSeries in econs:

    # get exporter network data
    # if country in net stats equals [0] <- exporter, [1] <- importers
    X_econ=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == tempSeries.split('-')[0]]]
    # network statistics availiable to exporter
    allNs=[col[1] for col in X_econ.columns]
    X_econ.columns=allNs

    # trade import series
    y=tsPctChange[[tempSeries]]
    y.columns = ['_'.join(col) for col in y.columns]

    for tempNs in allNs:

        X = X_econ[tempNs]
        X = sm.add_constant(X, has_constant='add')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        reg = sm.OLS(y_train, X_train).fit()

        y_pred = reg.predict(X_test)

        tempReturn=(pd.DataFrame({
            'ns':reg.params.index[1],
            'coef':reg.params[1],
            'pvalue':reg.pvalues[1],
            'r2':reg.rsquared,
            'aic':reg.aic,
            'mse':mean_squared_error(y_test, y_pred)},index=[tempSeries])
        )

        regResults.append(tempReturn)



regResults=pd.concat(regResults)
regResults.reset_index(inplace=True)


# %% [markdown]
#  ### 3. filter univariate regression results

# %%
regResults[regResults.index.isin(regResults['pvalue'].nsmallest().index)]



# %%
regResults[regResults.index.isin(regResults['r2'].nlargest().index)]



# %%
regResults[regResults.index.isin(regResults['aic'].nsmallest().index)]



# %%
regResults[regResults.index.isin(regResults['mse'].nsmallest().index)]



# %%
filteredRegResults=regResults.query('pvalue<0.05 and r2>.5')
filteredRegResults.reset_index(drop=True, inplace=True)
filteredRegResults

# %% [markdown]
#  ### 4. Visual Check

# %% [markdown]
#  Let's do a visual check of the series that came back with any remote form of a linear relationship.
# 
#  We can see that many of the relationships are affected by outliers so these numbers are misleading.
# 

# %%
from math import ceil


ncols=4

nrows = ceil(filteredRegResults.shape[0] / ncols)

width = ncols * 5
length = nrows * 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))

for i, ax in enumerate(axes.flatten()):
    if i < filteredRegResults.shape[0]:
        ax.scatter(
            x=tsPctChange[[filteredRegResults['index'][i]]],
            y=netStatsWidePctChange[[(f"{filteredRegResults['index'][i].split('-')[0]}", f"{filteredRegResults['ns'][i]}")]])

            # ax.suptitle(f"{filteredRegResults['index'][i]} Exports to {filteredRegResults['index'][i][1]} and {filteredRegResults['ns'][i]}")
        ax.set_title(f"pvalue:{np.round(filteredRegResults['pvalue'][i], 4)},  r2:{np.round(filteredRegResults['r2'][i], 2)},  aic:{np.round(filteredRegResults['aic'][i], 2)}")
        ax.set_ylabel(f"{filteredRegResults['ns'][i]} Percent Change")
        ax.set_xlabel(f"{filteredRegResults['index'][i]}  Percent Change")
    pass

plt.tight_layout()

# %% [markdown]
# ## 5. PCA on Network Statistics to Reduce Dimensionality

# %%

from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.linalg import eig

# %%
# %%
importers=pd.Series(col.split('-')[0] for col in tsPctChange.columns).unique()
exporters=pd.Series(col.split('-')[1] for col in tsPctChange.columns).unique()
allEcons=sorted(set(list(importers) + list(exporters)))


ncols=5
nrows = ceil(len(allEcons) / ncols)

width = ncols * 5
length = nrows * 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))

# for i, ax in enumerate(axes.flatten()):
def myplot(score,coeff, i, ax, tempSeries, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    ax.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        ax.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            ax.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    ax.set_title(tempSeries)


for i, econax in enumerate(zip(allEcons, axes.flatten())):

    tempSeries=econax[0]
    ax=econax[1]

    # tempSeries.split('-')[0] <- exporter, [1] <- importer
    temp=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == tempSeries]]
    # https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis

    if temp.shape[1] > 0:
        X = temp
        #In general a good idea is to scale the data
        scaler = StandardScaler()
        scaler.fit(X)
        X=scaler.transform(X)

        pca = PCA()
        x_new = pca.fit_transform(X)

        #Call the function. Use only the 2 PCs.
        myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]), i, ax, tempSeries, [col[1] for col in temp])

    plt.tight_layout()

# %% [markdown]
# Example of the Relationship between the original features and the principal components. The values can be interpreted as the correlation between the original feature and the component.

# %%

econ='Argentina'
temp=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == econ]]
scaler = StandardScaler()
scaledData = pd.DataFrame(scaler.fit_transform(temp))


pcaModel = PCA(n_components=3)


pcaModelFit = pcaModel.fit(scaledData)
principalComponents = pcaModelFit.transform(scaledData)

pcaModelFit.explained_variance_ratio_.sum()

loadings = pcaModelFit.components_.T * np.sqrt(pcaModelFit.explained_variance_)

loading_matrix = pd.DataFrame(loadings, index=temp.columns)
print(pcaModelFit.explained_variance_ratio_.sum())
loading_matrix.sort_values(by=[0], ascending=False)



# %% [markdown]
# an attempt to interpret the principal components:
# 
# PC 0: Centrality/Degree measures -> "Connectivity"
# 
# PC 1: Macro features such as number of edges and nodes while negatively related to pagerank values 
# 
# PC 2: A bit of everything, overlaps pagerank and number of edges/nodes which are clearly seperated in PC 1

# %%

econs=pd.Series(col for col in tsPctChange.columns).unique()
regResultsPCA=[]
for tempSeries in econs:

    # network statistics for reference econ
    X_econ=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == tempSeries.split('-')[0]]]

    # if there is data
    if X_econ.shape[1] > 0:

        # need to allNs for later
        allNs=[col[1] for col in X_econ.columns]
        X_econ.columns=allNs

        scaler = StandardScaler()
        scaledData = pd.DataFrame(scaler.fit_transform(X_econ))

        #####   PCA
        # create model
        n_components=3
        pcaModel = PCA(n_components=n_components)

        # fit model
        pcaModelFit = pcaModel.fit(scaledData)
        X_econ = pd.DataFrame(pcaModelFit.transform(scaledData), columns=[str(col) for col in range(n_components)])

        # trade time series for reference econ
        y=tsPctChange[[tempSeries]]
        y.columns = ['_'.join(col) for col in y.columns]

        X_econ.index=y.index

        for tempNs in X_econ.columns:
            # if tempNs in X.columns:
            X = X_econ[tempNs]
            X = sm.add_constant(X, has_constant='add')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

            reg = sm.OLS(y_train, X_train).fit()

            y_pred = reg.predict(X_test)


            tempReturn=(pd.DataFrame({
                'ns':reg.params.index[1],
                'coef':reg.params[1],
                'pvalue':reg.pvalues[1],
                'r2':reg.rsquared,
                'aic':reg.aic,
                'mse':mean_squared_error(y_test, y_pred)},index=[tempSeries])
            )

            regResultsPCA.append(tempReturn)

regResultsPCA=pd.concat(regResultsPCA)
regResultsPCA.reset_index(inplace=True)



# %%
regResultsPCA[regResultsPCA.index.isin(regResultsPCA['pvalue'].nsmallest().index)]



# %%
regResultsPCA[regResultsPCA.index.isin(regResultsPCA['r2'].nlargest().index)]



# %%
regResultsPCA[regResultsPCA.index.isin(regResultsPCA['aic'].nsmallest().index)]



# %%
regResultsPCA[regResultsPCA.index.isin(regResultsPCA['mse'].nsmallest().index)]



# %%
regResultsPCA[regResultsPCA.index.isin(abs(regResultsPCA['coef']).nlargest().index)]



# %%
filteredregResultsPCA=regResultsPCA.query('pvalue<0.1 and r2>0.2')
filteredregResultsPCA.reset_index(drop=True, inplace=True)
filteredregResultsPCA

# %%
from math import ceil


ncols=4

nrows = ceil(filteredregResultsPCA.shape[0] / ncols)

width = ncols * 5
length = nrows * 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))

for i, ax in enumerate(axes.flatten()):
    if i < filteredregResultsPCA.shape[0]:
        econ=filteredregResultsPCA['index'][i].split('-')[0]
        temp=netStatsWidePctChange[[col for col in netStatsWidePctChange.columns if col[0] == econ]]
        if temp.shape[1] > 0:
            scaler = StandardScaler()
            scaledData = pd.DataFrame(scaler.fit_transform(temp))

            #####   PCA
            # create model
            pcaModel = PCA(n_components=3)

            # fit model
            pcaModelFit = pcaModel.fit(scaledData)
            principalComponents = pcaModelFit.transform(scaledData)

            ax.scatter(
                x=tsPctChange[[filteredregResultsPCA['index'][i]]],
                #y=netStatsWidePctChange[[(f"{filteredregResultsPCA['index'][i][0]}", f"{filteredregResultsPCA['ns'][i]}")]])
                y=pd.DataFrame(principalComponents)[int(filteredregResultsPCA['ns'][i])]
                )

            # ax.set_suptitle(f"{filteredregResultsPCA['index'][i][0]} Exports to {filteredregResultsPCA['index'][i][1]} and {filteredregResultsPCA['ns'][i]}")
            ax.set_title(f"pvalue:{np.round(filteredregResultsPCA['pvalue'][i], 5)},  r2:{np.round(filteredregResultsPCA['r2'][i], 2)},  aic:{np.round(filteredregResultsPCA['aic'][i], 2)}")
            ax.set_ylabel(f"{filteredregResultsPCA['ns'][i]} Percent Change")
            ax.set_xlabel(f"{filteredregResultsPCA['index'][i]}")

plt.tight_layout()



