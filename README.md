# Trade Networks

Analysis of international trade networks.

IMF's Direction of Trade Statistics (DOTS) is used.

- information extraction from network
    - multi-step forcasting with network statistics
        - linear regression
        - xgboost

Data is queried from dbnomics api and stored locally in csv's.

DOTS metadata only availiable in R from dbnomics.

Terminology:

    TXG_FOB_USD, export series:
        reference economy exports to partner economy
        uses "Freight on Board" price
        currently used in project
    TMG_CIF_USD, import series:
        partner economy exports to reference economy
        uses "Cost, Insurance, and Freight" price


TODO:
- link level features calculation -> linkFeatures.py
- better experiment tracking / performance comparison
- expanding and sliding window time series
- link prediction ~ network structure forecasting
- make better use of parameters in ploomber
- dockerize
-
# Files

    .
    ├── 00-data  
    │   ├── downLoadMetaData.R          1. downloads DOTS metadata from dbnomics
    │   ├── dataCollect.py              2. downloads DOTS data from dbnomics
    │   ├── createTimeSeries.py         3. creates dotsTimeSeries.csv
    │   ├── calculateNetworkStats.py    4. creates DOTSnetStats.csv
    │   ├── linkFeatures.py             5. creates linkFeatures.csv
    │   ├── metadata                    6. metadata for api queries
    │   │   ├── countries.csv           7. reference economies
    │   │   └── counterparts.csv        8. partner economies
    │   ├── raw                         9. dump for raw data
    │   │   └── ...                     10. individual economies' csvs
    │   └──  clean                      11. dump for clean data
    │       ├── DOTS.csv                12. aggregated DOTS dataset
    │       ├── dotsTimeSeries.csv      13. filtered econ's with many Nan's
    │       ├── dotsTimeSeriesAll.csv   1 . all econ's
    │       ├── DOTSnetStats.csv        1 . calculated network statistics
    │       └── ...                     1 . ploomber .ipynb and .metadata files
    └── 01- time series . 
        ├── linearRegression            1 . univar + PCA, 90/10 split
        ├── xgboostWindow.py            1 . multivar & 5 lags of y, 90/10 split
        ├── xgboostDirection.py         1 . multivar & 5 lags of y, 90/10 split
        └── xgboostDOTS.py              1 . multivar + PCA,  90/10 split

## File Descriptions


- 3 creatTimeSeries.py  
    Creates a long dataset of all bilateral trade series.
    Creates another long dataset of trade series without missing data.  
    Steps for the filtered dataset:  
    - started with x exporters
    - filter period to start at 1980
    - remove 10 importers with most missing data across all series
    - remove any remaining series with nan value
    - ended with 50 exporters and x importers


- 17 xgboostWindow.py  
    uses 5 lags of the dependent variable, and node features for both the importer and exporter. It also uses a sliding window of 12 months.

- 17 xgboostDirection.py  
    uses 5 lags of the dependent variable, and node features for both the importer and exporter. It also uses a sliding window of 12 months. 
    Goal is to predict direction of next step

- 18 xgboostDOTS.py  
    uses node features for importer and exporter, trains on first 90% and tests on last 10%



# References

- [Using networks for link prediction](https://arxiv.org/pdf/2110.11751.pdf)


# Setup

```sh
# activate environment (windows cmd.exe)
{C:\Users\yosty\Envs\dotsPloomber}
{path-to-venv}\Scripts\activate.bat

```

## Code editor integration

* If using Jupyter, [click here](https://docs.ploomber.io/en/latest/user-guide/jupyter.html)
* If using VSCode, PyCharm, or Spyder, [click here](https://docs.ploomber.io/en/latest/user-guide/editors.html)



## Running the pipeline

```sh
ploomber nb -i
ploomber build
```

## Help

* Need help? [Ask us anything on Slack!](https://ploomber.io/community)