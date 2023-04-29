import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def parser(x):
    return datetime.strptime(x, '%d-%m-%Y')


def plot_seasonal_decompose(df_serie, period):
    decomposition = seasonal_decompose(df_serie, period=period)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(df_serie, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()


def _plot_rolling_statistics(timeseries, window_size):
    rolmean = timeseries.rolling(window=window_size).mean()
    rolstd = timeseries.rolling(window=window_size).std()

    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


def _test_dickey_fuller(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used',
                                'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def test_stationarity(timeseries, window_size):
    _plot_rolling_statistics(timeseries, window_size)
    _test_dickey_fuller(timeseries)


def fill_zeros_with_mean_per_day(df):
    df['date_visit'] = pd.to_datetime(df['date_visit'])
    df['Dia_semana'] = df['date_visit'].dt.dayofweek
    df_sin_ceros = df[df['visits'] != 0]
    medias_visitas = df_sin_ceros.groupby('Dia_semana')['visits'].mean()

    for index, row in df.iterrows():
        if row['visits'] == 0:
            dia_semana = row['Dia_semana']
            media_dia = medias_visitas[dia_semana]
            df.at[index, 'visits'] = media_dia

    return df
