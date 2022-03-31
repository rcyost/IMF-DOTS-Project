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
upstream = ['downLoadMetaData'] # this means: execute raw.py, then clean.py
product = None


# %% tags=["injected-parameters"]
# Parameters
upstream = {"downLoadMetaData": {"nb": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\downLoadMetaData.ipynb", "counterparts": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\metadata\\counterparts.csv", "countries": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\metadata\\countries.csv"}}
product = {"nb": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\dataCollect.ipynb", "files": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\raw", "DOTS": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\clean\\DOTS.csv"}


# %%

import pandas as pd
from dbnomics import fetch_series
import os

# r seems to have saved this in some odd way
countries=pd.read_csv(upstream['downLoadMetaData']['countries'])
counterparts=pd.read_csv(upstream['downLoadMetaData']['counterparts'])


# %%
# TODO: parameterize import/export with ploomber
# goods, value of imports CIF price - TMG_CIF_USD
# goods, value of exports FOD - TXG_FOB_USD


for econ in countries['REF_AREA']:
    try:
        data = fetch_series(
                provider_code='IMF',
                dataset_code='DOT',
                max_nb_series=1000000,
                dimensions={
                "FREQ":["M"],
                "REF_AREA":[f"{econ}"],
                "INDICATOR":["TXG_FOB_USD"]}
            )
        data.to_csv(rf'{str(product["files"])}\{econ}.csv')
    except:
        # TODO: Log this economy that didn't work
        pass

# %%

path = product['files']
directory = os.fsencode(path)

dfList = []

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".csv"):
         dfList.append(pd.read_csv(os.path.join(path, filename)))
     else:
         continue

data = pd.concat(dfList)

data.to_csv(product['DOTS'])
