
# this took 9 hours
#%%

import pandas as pd
import os
import requests
import time_series_datapoints_df as wto

#%%
keys = ['9700b543609e4af7b8f05832aa858ae0',
            '6a77840f716f422eb8d2b25d06d3da23',
            '221f238346524ae4a876d57598c3afd9',
            '6d170f885df24a858c2aca20a0914911']

def get_products(key,
                 name=None,
                 product_classification='all',
                 lang=1,
                 proxies=None):
    if name is None:
        endpoint = f'https://api.wto.org/timeseries/v1/products?pc={product_classification}&lang={lang}' \
                     f'&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data
    else:
        endpoint = f'https://api.wto.org/timeseries/v1/products?name={name}&pc={product_classification}' \
                   f'&lang={lang}&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data

def get_topics(
        key,
        lang=1,
        proxies=None):
    endpoint = f'https://api.wto.org/timeseries/v1/topics?lang={lang}&subscription-key={key}'
    response = requests.get(endpoint, proxies=proxies)
    returnedData = response.json()

    #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

    # ok so this has been done one one line but at what cost
    data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

    return data


def get_indicators(key,
                   indicator_code='all',
                   name=None,
                   topics='all',
                   product_classification='all',
                   trade_partner='all',
                   frequency='all',
                   lang=1,
                   proxies=None):
    if name is None:
        endpoint = f'https://api.wto.org/timeseries/v1/indicators?i={indicator_code}&t={topics}' \
                   f'&pc={product_classification}&tp={trade_partner}&frq={frequency}&lang={lang}' \
                   f'&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data
    else:
        endpoint = f'https://api.wto.org/timeseries/v1/indicators?i={indicator_code}&name={name}&t={topics}' \
                   f'&pc={product_classification}&tp={trade_partner}&frq={frequency}&lang={lang}' \
                   f'&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data

#%%
key=keys[0]

indicators = get_indicators(key=key)
# reportEcon = wto.get_reporting_economies(key=key)
# products = wto.get_products(key=key)
products = get_products(key=key)
topics=get_topics(key=key)

#%%

