# -*- coding: cp949 -*-
import FinanceDataReader as fdr
from datetime import date
import pandas as pd
import streamlit as st
from plot_tools import bokeh_multiline


def pct_change(x):
    first, last = x.iloc[0], x.iloc[-1]
    # if x.size > 40:
    #     first, last = np.mean(x.iloc[0:5]), np.mean(x.iloc[-6:-1])
    return (last-first) / first


def is_unrate_increased_to_point(x):
    """
    (1년)전시점과 현재시점의 실업율 비교
    """
    first, last = x.iloc[0], x.iloc[-1]
    return first < last


def is_unrate_increased_to_mean(x):
    """
    최근 (1년)의 평균과 현재시점의 실업율 비교
    """

    first, last = x.iloc[0], x.iloc[-1]
    period_mean = x.mean()
    return period_mean < last


def period_weighted_momentums(series, rolling_param):
    """
    :param series:
    :param rolling_param:
        periods =[30D, 91D, 182D, 365D]
        multipliers = [6, 4, 2, 1]
        => sum(periods * multipliers)
    :return:
    """

    rolling_periods, multipliers = rolling_param['periods'], rolling_param['multipliers']

    momentums = [series.rolling(period).apply(pct_change) for period in rolling_periods]
    df_momentums = pd.DataFrame({name:base for name, base in zip(rolling_periods, momentums)})
    df_momentums['sum'] = sum(df_momentums[name] * mul for name, mul in zip(rolling_periods, multipliers))
    return df_momentums


def keller_is_bull_market(is_unrate_increase, momentums):
    echo = False
    is_all_positive_km = all(m > 0 for m in momentums)
    if echo: print('===== Canary Indicator =====')
    if is_unrate_increase:
        if echo: print('1. 작년에 비해 실업율이 높음 --> VWO/BND 확인')

        if is_all_positive_km:
            if echo: print('2. VWO/BND 모두 양수 --> 주식에 투자')
            return True
        else:
            if echo: print('2. VWO/BND 모두 양수 아님 --> 채권에 투자')
            return False
    else:
        if echo: print('1. 작년에 비해 실업율이 낮음 --> 주식에 투자')
        return True

@st.cache
def data_from_csv():
    df_km_prices = pd.read_csv('data/keller_index.csv', header=0, index_col='Date', parse_dates=['Date'])
    unrate = pd.read_csv('data/unrate.csv', header=0, index_col='DATE', parse_dates=['DATE'])
    spy = pd.read_csv('data/spy.csv', header=0, index_col='Date', parse_dates=['Date'])
    return df_km_prices, unrate, spy


def data_from_fdr():
    save_to_file, start_date = True, None

    km_tickers = 'BND VWO'.split()
    km_list = [fdr.DataReader(ticker, start=start_date)['Close'] for ticker in km_tickers]
    df_km_prices = pd.concat(km_list, axis=1)
    df_km_prices.columns = km_tickers

    unrate = fdr.DataReader('UNRATE', start=start_date, data_source='fred')
    spy = pd.DataFrame({'SPY': fdr.DataReader('SPY')['Close']})
    spy.index.name = 'Date'

    if save_to_file:
        df_km_prices.to_csv('data/keller_index.csv', mode='w+')
        unrate.to_csv('data/unrate.csv', mode='w+')
        spy.to_csv('data/spy.csv', mode='w+')

    return df_km_prices, unrate, spy


@st.cache
def period_weighted_momentum_indicator(df_km, unrate, rolling_param):

    df_momentums = [period_weighted_momentums(df_km[item], rolling_param) for item in df_km.columns]

    indicators = {name+'_M': base['sum'] for name, base in zip(df_km.columns, df_momentums)}
    indicators['UNRATE'] = unrate.asfreq(
        'D', method='ffill')['UNRATE'].rolling('365D').apply(is_unrate_increased_to_point)

    df_indicator = pd.DataFrame(indicators)
    df_indicator['UNRATE'] = df_indicator['UNRATE'].fillna(method='ffill')
    df_indicator = df_indicator.dropna()

    df_indicator['bull_market'] = df_indicator.apply(lambda x: int(keller_is_bull_market(x['UNRATE'], [x[name+'_M'] for name in df_km.columns])), axis=1)

    return df_indicator


def ma_is_bull_market(is_unrate_increased, above_ma):
    if is_unrate_increased:
        return True if above_ma else False
    else:
        return True

@st.cache
def moving_average_indicator(stock_data, unrate):
    df = stock_data.copy()
    c_name = df.columns[0]
    df['200MA'] = df[c_name].rolling('200D').mean()
    df['UNRATE'] = unrate.asfreq('D', method='ffill')['UNRATE'].rolling('365D').apply(is_unrate_increased_to_mean)

    df['UNRATE'] = df['UNRATE'].fillna(method='ffill')
    df = df.dropna()

    df['bull_market'] = df.apply(lambda x: int(ma_is_bull_market(x['UNRATE'], x[c_name] > x['200MA'])), axis=1)
    return df


def indicator_plot(df, start, end, title, secondary_y=False):
    df_part = df.loc[start:end].copy()
    df_part['UNRATE'] = df_part['UNRATE'].replace([0], -1)
    df_part['bull_market'] = df_part['bull_market'].replace([0], -1) * 1.5

    if secondary_y:
        st.subheader(title)
        p = bokeh_multiline(df_part, ['UNRATE', 'bull_market'])
        st.bokeh_chart(p, use_container_width=True)
    else:
        # ax = df_part.plot(title=title)
        # ax.axhline(y=0, color='y', linestyle='--', lw=1)
        st.subheader(title)
        st.line_chart(df_part)


def indicator_comparison_plot(perf_indicators, perf_references, start, end):
    delta = 0.2
    ref = {name: series.replace([0], -1) * (delta * i +1) for i, (name, series) in enumerate(perf_indicators.items())}
    st.subheader('Indicator Comparison')
    df = pd.DataFrame({**ref, **perf_references}).loc[start:end]
    p = bokeh_multiline(df, perf_indicators.keys())
    st.bokeh_chart(p, use_container_width=True)
    return p


def app():

    rolling_param_default = {'periods': '30D, 91D, 182D, 365D', 'multipliers': '6, 4, 2, 1'}

    st.sidebar.subheader('Trading Indicators')
    source = st.sidebar.radio("Source of data", ('Downloaded', 'Web crawling'))
    if source == 'Web crawling':
        km_pirces, unrate, spy = data_from_fdr()
    elif source == 'Downloaded':
        km_pirces, unrate, spy = data_from_csv()
    else:
        print('Error!')

    st.sidebar.subheader('Period Weighted Momentum(PWM)')
    periods = st.sidebar.text_input('Periods', value=rolling_param_default['periods'])
    multipliers = st.sidebar.text_input('Multipliers', value=rolling_param_default['multipliers'])

    st.sidebar.subheader('Charts')
    start = st.sidebar.date_input('Start date', date(2019, 7, 6))
    end = st.sidebar.date_input('End date', date.today())

    rolling_param = {'periods': periods.split(','), 'multipliers': list(map(int, multipliers.split(',')))}

    keller_pwm_indicator = period_weighted_momentum_indicator(km_pirces, unrate, rolling_param)
    spy_ma_indicator = moving_average_indicator(spy, unrate)
    keller_spy_indicator = period_weighted_momentum_indicator(spy, unrate, rolling_param)

    perf_indicators = {'Keller Period Weighted Momentum': keller_pwm_indicator['bull_market'],
                       'Moving Average(spy)': spy_ma_indicator['bull_market'],
                       'Keller PWM(spy)': keller_spy_indicator['bull_market']}
    perf_references = {'SPY': spy['SPY'], 'SPY200MA': spy['SPY'].rolling('200D').mean()}

    if st.sidebar.checkbox('Plot indicators', value=True):
        indicator_plot(keller_pwm_indicator, start, end, 'Keller Period Weighted Momentum')
        indicator_plot(spy_ma_indicator, start, end, 'Moving Average(SPY, 200days)', secondary_y=True)
        indicator_plot(keller_spy_indicator, start, end, 'Keller PWM(using SPY ticker)')

    if st.sidebar.checkbox('Indicator comparison'):
        indicator_comparison_plot(perf_indicators, perf_references, start, end)


if __name__ == '__main__':
    app()

