# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# %% tags=["parameters"]
upstream = ['dataCollect']
product = None


# %% tags=["injected-parameters"]
# Parameters
upstream = {"dataCollect": {"nb": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\dataCollect.ipynb", "files": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\raw", "DOTS": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\clean\\DOTS.csv"}}
product = {"nb": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\createTimeSeries.ipynb", "dotsTimeSeries": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\clean\\dotsTimeSeries.csv"}


# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#%%[markdown]

# Load in Data
# Remove groups of economies

# %%

rawData=pd.read_csv(upstream['dataCollect']['DOTS'])
keepCols=['series_code', 'period', 'value', 'REF_AREA', 'INDICATOR', 'COUNTERPART_AREA', 'Reference Area', 'Counterpart Reference Area']
cleanData = rawData[keepCols]
cleanData.rename(columns={'Counterpart Reference Area':'CounterpartReferenceArea',
        'Reference Area':'ReferenceArea'}, inplace=True)
wideData = (cleanData
    .pivot_table(
        index=['period', 'CounterpartReferenceArea'],
        columns='ReferenceArea',
        values='value'))

# remove groups
# decided to keep 'Special Categories and Economic Zones' - sounds interesting
# and wouldn't cause overlap to include i think
removeList = [
'Export earnings: fuel',
'Export earnings: nonfuel',
'Middle East, North Africa, Afghanistan, and Pakistan',
'Emerging and Developing Europe',
'Western Hemisphere',
'Western Hemisphere not allocated',
'EU (Member States and Institutions of the European Union) changing composition',
'Euro Area (Member States and Institutions of the Euro Area) changing composition',
'Europe',
'Europe not allocated',
'Africa',
'Africa not allocated',
'Sub-Saharan Africa',
'Middle East',
'Middle East and Central Asia not specified',
'Other Countries n.i.e. (IMF)',
'Advanced Economies (IMF)',
'Emerging and Developing Countries',
'Developing Asia (IMF)',
'Middle East and Central Asia',
'Belgo-Luxembourg Economic Union',
'Community of Independent States (CIS)',
'Asia not allocated',
'Former U.S.S.R.',
'All Countries, excluding the IO']

dataLong = (wideData
    .melt(ignore_index=False)
    .query('CounterpartReferenceArea not in @removeList')
    .query('ReferenceArea not in @removeList')
)


dataLong.reset_index(inplace=True)
dataLong['period'] = pd.to_datetime(dataLong['period'])

wideData2 = dataLong.pivot_table(index=['period', 'CounterpartReferenceArea'],
                                columns='ReferenceArea',
                                values='value')

dataLong.to_csv(product['dotsTimeSeriesAll'])


#%%[markdown]

# Missing Data per Exporter



# %%

# nanData=dataLong['value'].isna().groupby([dataLong['period'], dataLong['ReferenceArea']]).sum()

# nanData=pd.DataFrame(nanData).reset_index()

# nanData=nanData.pivot_table(index='period', columns='ReferenceArea', values='value')

# #nanData=nanData[nanData.index > "1990-01-01"]

# nanData=nanData / dataLong['CounterpartReferenceArea'].nunique()


# inputData=nanData
# ncols=2
# width=10
# length=100

# from math import ceil

# nrows = ceil(len(inputData.columns) / ncols)

# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))
# for i, ax in enumerate(axes.flatten()):
#     if i > len(inputData.columns):
#         pass
#     else:
#         data = inputData[inputData.columns[i]]
#         ax.plot(data, color='red', linewidth=1)
#         # Decorations
#         ax.set_title(inputData.columns[i])
#         ax.xaxis.set_ticks_position('none')
#         ax.get_xaxis().set_visible(False)
#         ax.yaxis.set_ticks_position('none')
#         ax.spines["top"].set_alpha(0)
#         ax.tick_params(labelsize=6)

# plt.tight_layout()


#%%[markdown]

# 10 Importers with most missing data across all bilateral series


# %%

################################################## 10 exporters causing the most missing data

# amount of missing data by year, exporter
nanData=dataLong['value'].isna().groupby([dataLong['period'], dataLong['ReferenceArea']]).sum()
nanData=pd.DataFrame(nanData).reset_index()
nanData=nanData.pivot_table(index='period', columns='ReferenceArea', values='value')

# total missing
totalNans=nanData.values.sum()
# upper bound?
totalCombo=dataLong['CounterpartReferenceArea'].nunique() * dataLong['ReferenceArea'].nunique()

results={}
for col in nanData:

    # amount of missing data caused by economy in loop iteration
    temp=dataLong[dataLong['ReferenceArea']!=col]
    tempNanData=temp['value'].isna().groupby([dataLong['period'], dataLong['ReferenceArea']]).sum()
    results[col] = totalNans-tempNanData.values.sum()
    # print(col, ' : ',totalNans-tempNanData.values.sum())

tenLargestMissing=pd.DataFrame(sorted(results.items(), key=lambda item: item[1])).tail(10)
removeEcons=[econ for econ in tenLargestMissing[0]]


#%%[markdown]

# 10 Importers with most missing data across all bilateral series


# %%

# nanData=dataLong['value'].isna().groupby([dataLong['period'], dataLong['ReferenceArea']]).sum()

# nanData=pd.DataFrame(nanData).reset_index()

# nanData=nanData.pivot_table(index='period', columns='ReferenceArea', values='value')

# nanData=nanData / dataLong['CounterpartReferenceArea'].nunique()

# for col in nanData:
#     # https://stackoverflow.com/a/49573439/11706269
#     plt.plot(nanData[[col]])
#     ax=plt.gca()
#     ax.xaxis.set_major_locator(mdates.YearLocator(10))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ax.set_title('percent of data missing')

# %%

largestMissing=pd.DataFrame(sorted(results.items(), key=lambda item: item[1])).tail(100)
removeEcons=[econ for econ in largestMissing[0]]
# ~ is the not operator, I would have guessed !
tempLong=dataLong[~dataLong['CounterpartReferenceArea'].isin(removeEcons)]
tempLong=tempLong[~tempLong['ReferenceArea'].isin(removeEcons)]

# %%

# dates=tempLong['period'].unique()
# droppedShapes=[]
# for date in dates:

#     temp=tempLong[tempLong['period'] > date]
#     temp=temp.pivot_table(index='period', columns=['ReferenceArea', 'CounterpartReferenceArea'], values='value')
#     temp=temp.dropna(axis=1)
#     print(temp.shape)
#     droppedShapes.append(pd.DataFrame({'date':date, 'length': temp.shape[0], 'series':temp.shape[1]}, index=[date]))

# dropDf=pd.concat(droppedShapes)
# dropDf.plot.scatter(x='length', y='series')

# %%

temp=tempLong[tempLong['period'] > '1980-01-01']
temp=temp.pivot_table(index='period', columns=['ReferenceArea', 'CounterpartReferenceArea'], values='value')
temp.dropna(axis=1, inplace=True)

# %%

# for col in temp:
#     # https://stackoverflow.com/a/49573439/11706269
#     plt.plot(temp.index, temp[[col]])
#     ax=plt.gca()
#     ax.xaxis.set_major_locator(mdates.YearLocator(10))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# %%

temp=temp.melt(ignore_index=False)
temp.to_csv(product['dotsTimeSeries'])
