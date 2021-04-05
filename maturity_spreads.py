# -*- coding: cp949 -*-

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import datapackage
from functools import reduce
from datetime import date
import streamlit as st
from plot_tools import bokeh_multiline


def from_datahub(url):
    package = datapackage.Package(url)
    resources = package.resources
    for resource in resources:
        if resource.tabular:
            data = pd.read_csv(resource.descriptor['path'])

    data['DATE'] = pd.to_datetime(data['Date'])
    data.set_index('DATE', inplace=True)
    data.drop(['Date'], axis=1, inplace=True)

    return data


def data_from_datahub():
    save_to_file = True

    sp500_url = 'https://datahub.io/core/s-and-p-500/datapackage.json'
    gold_url = 'https://datahub.io/core/gold-prices/datapackage.json'

    sp500 = from_datahub(sp500_url)
    sp500 = sp500[['SP500']]
    sp500.columns = ['SP500']

    gold = from_datahub(gold_url)
    gold.columns = ['GOLD']


def data_from_csv():
    us_2y_spread = pd.read_csv('data/us_2y_spread.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    us_3m_spread = pd.read_csv('data/us_3m_spread.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    nasdaqcom = pd.read_csv('data/nasdaqcom.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    houst = pd.read_csv('data/houst.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    gold = pd.read_csv('data/gold.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    sp500 = pd.read_csv('data/sp500.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    dgs10 = pd.read_csv('data/dgs10.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    dgs2 = pd.read_csv('data/dgs2.csv', header=0, index_col='DATE', parse_dates=['DATE'])

    return [nasdaqcom, houst, gold, sp500, dgs10, dgs2, us_2y_spread, us_3m_spread]


def data_from_web():
    save_to_file = True

    # FRED 주요 통계 -장단기금리차 (10-Year Treasury Constant Maturity Minus 2-Year Treasury  Constant Maturity)
    us_2y_spread = fdr.DataReader('T10Y2Y', data_source='fred')
    us_3m_spread = fdr.DataReader('T10Y3M', data_source='fred')
    dgs10 = fdr.DataReader('DGS10', data_source='fred')
    dgs2 = fdr.DataReader('DGS2', data_source='fred')

    nasdaqcom = fdr.DataReader('NASDAQCOM', data_source='fred')  # NASDAQCOM 나스닥종합지수
    houst = fdr.DataReader('HOUST', data_source='fred')  # HOUST: Housing Starts: Total: New Privately Owned Housing Units Started

    sp500_url = 'https://datahub.io/core/s-and-p-500/datapackage.json'
    gold_url = 'https://datahub.io/core/gold-prices/datapackage.json'

    sp500 = from_datahub(sp500_url)
    sp500 = sp500[['SP500']]
    sp500.columns = ['SP500']

    gold = from_datahub(gold_url)
    gold.columns = ['GOLD']

    if save_to_file:
        us_2y_spread.to_csv('data/us_2y_spread.csv', mode='w+')
        us_3m_spread.to_csv('data/us_3m_spread.csv', mode='w+')
        nasdaqcom.to_csv('data/nasdaqcom.csv', mode='w+')
        houst.to_csv('data/houst.csv', mode='w+')
        gold.to_csv('data/gold.csv', mode='w+')
        sp500.to_csv('data/sp500.csv', mode='w+')
        dgs10.to_csv('data/dgs10.csv', mode='w+')
        dgs2.to_csv('data/dgs2.csv', mode='w+')

    return [nasdaqcom, houst, gold, sp500, dgs10, dgs2, us_2y_spread, us_3m_spread]


def app():

    st.sidebar.subheader('Maturity spread')
    source = st.sidebar.radio("Source of data", ('Downloaded', 'Web crawling'))
    if source == 'Web crawling':
        dfs = data_from_web()
    elif source == 'Downloaded':
        dfs = data_from_csv()
    else:
        print('Error!')

    st.sidebar.subheader('Charts')
    start = st.sidebar.date_input('Start date', date(1980, 1, 1))
    end = st.sidebar.date_input('End date', date.today())

    merged_df = reduce(lambda left, right: left.join(right, how='outer'), dfs).fillna(method='ffill')
    print(merged_df.dtypes)
    merged_df['S2Y'] = np.where(merged_df['T10Y2Y'] < 0, -1, np.where(merged_df['T10Y2Y'] > 0, 1, 0))
    merged_df['S3M'] = np.where(merged_df['T10Y3M'] < 0, -1, np.where(merged_df['T10Y3M'] > 0, 1, 0))

    merged_df = merged_df.loc[start:end]
    columns = np.array(merged_df.columns)
    filters = np.array([True] * (len(columns)))
    for i, c in enumerate(columns):
        filters[i] = st.sidebar.checkbox(columns[i], value=filters[i])

    #T10Y2Y  T10Y3M  NASDAQCOM   HOUST    GOLD  SP500

    print(merged_df.head())
    filtered_cols = columns[filters]
    merged_df = merged_df[filtered_cols]

    p = bokeh_multiline(merged_df, set(filtered_cols).intersection(['DGS10', 'DGS2', 'T10Y2Y', 'T10Y3M', 'S2Y', 'S3M']))
    st.bokeh_chart(p, use_container_width=True)


    # ax = merged_df.plot(secondary_y=['T10Y2Y', 'T10Y3M'])
    # ax.right_ax.axhline(y=0, color="red", alpha=0.5)
    #
    # plt.show()


if __name__ == '__main__':
    app()