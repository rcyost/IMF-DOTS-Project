


import numpy as np
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import concurrent

import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

import time_series_datapoints_df as wto

import logging

from  dbnomics import fetch_series

import logging
from multiprocessing.pool import Pool
import time_series_datapoints_df as wto


import logging

# this took 9 hours

# TODO: logging properly

def setup_ITS_MTV_AX(key):

    # http://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2017-edition/hs-nomenclature-2017-edition.aspx

    # only required field other than key
    indicators = wto.get_indicators(key=key)
    reportEcon = wto.get_reporting_economies(key=key)
    products = wto.get_products(key=key)

    # look for datasets
    code = "HS_M"
    of_interest = indicators[indicators.code.str.contains(code)]

    # sitc3 codes
    # codes = ["31", "32", "33"]
    codes = ["30"]
    queryProduct = products.query('hierarchy == @codes & productClassification == "SITC3"')
    # extract productSector codes in a string seperated by commas to query with

    ps_query_string = ''
    for code in queryProduct.code:
        ps_query_string = ps_query_string + code + ','

    # this adds extra comma for last code so we remove
    ps_query_string = ps_query_string[:-1]

    reportEcon_string = ''
    for code in reportEcon.code:
        reportEcon_string = reportEcon_string + code + ','

    # this adds extra comma for last code so we remove
    reportEcon_string = reportEcon_string[:-1]

    return ps_query_string, reportEcon_string

def query_ITS(key_str:str, years:list):

    ps_query_string, reportEconString = setup_ITS_MTV_AX(key_str)


    logging.basicConfig(filename=f'log{years[0]}-{years[1]}.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')


    listOfData = []

    #indicators = ['ITS_MTV_AX', 'ITS_MTV_AM', 'ITS_CS_QAX', 'ITS_CS_QAM']
    #indicators = 'HS_M_0010'
    #indicators = 'ITS_MTV_AM'
    # indicators = 'ITS_MTV_AM'
    # indicators = 'ITS_MTV_MX'
    indicators = 'ITS_MTV_MM'

    okList = []
    failList = []
    fail = 0
    ok = 0

    for year in range(years[0], years[1]):

        try:
            data = wto.get_time_series_datapoints(
                                                indicator_code=indicators,
                                                reporting_economy=reportEconString,
                                                product_sector=ps_query_string,
                                                time_period = year,
                                                max_records=1000000, # max is 1 million
                                                key=key_str)
            #listOfData.append(data)
            data.to_csv(rf'{str(product["ism"])}\{indicators}\{year}.csv')

            print(f'ok: {year} {key_str}')
            logging.info(f'ok: {year} {key_str}')

        except:

            print(f'fail: {year} {key_str}')
            logging.info(f'fail: {year} {key_str}')

    print(f'COMPLETE -------------- {years[0]} {years[1]} {key_str}')


keys = ['9700b543609e4af7b8f05832aa858ae0',
            '6a77840f716f422eb8d2b25d06d3da23',
            '221f238346524ae4a876d57598c3afd9',
            '6d170f885df24a858c2aca20a0914911']

# years = [[1996, 2003],
#             [2003, 2010],
#             [2010, 2017],
#             [2017, 2021]]

years = [[2006, 2010],
            [2010, 2013],
            [2013, 2017],
            [2017, 2021]]

# years = [[2002, 2003],[2006, 2007], [2016, 2017]]

if __name__ == '__main__':
    with Pool(4) as p:
        p.starmap(query_ITS, [z for z in zip(keys, years)])


