from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def asset_ts_from_df(df, asset, freq='30min'):
    data=df[df['asset1']==asset]
    return data.set_index('time_5min')[['algo_price', 'algo_volume']].asfreq(freq)

def get_mean_std(data, window):
    return data.rolling(window).mean(), data.rolling(window).std()

#https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
def next_step_MA(data, key='algo_price'):
    '''
    Moving Average. 
    Univariate time series without trend and seasonal components.
    '''
    model = ARIMA(data[key], order=(0, 0, 1))
    model_fit = model.fit()
    return model_fit.predict(len(data), len(data))

def next_step_ARMA(data, key='algo_price'):
    '''
    Autoregressive Moving Average.
    Univariate time series without trend and seasonal components.
    '''
    model = ARIMA(data[key], order=(2, 0, 1))
    model_fit = model.fit()
    return model_fit.predict(len(data), len(data))

def next_step_ARIMA(data, key='algo_price'):
    '''
    Autoregressive Integrated Moving Average.
    Univariate time series with trend and without seasonal components.
    '''
    model = ARIMA(data[key], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit.predict(len(data), len(data), typ='levels')

def next_step_SARIMA(data, key='algo_price'):
    '''
    Seasonal Autoregressive Integrated Moving-Average. 
    Univariate time series with trend and/or seasonal components.
    '''
    model = SARIMAX(data[key], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    return model_fit.predict(len(data), len(data))

def next_step_SES(data, key='algo_price'):
    '''
    Simple Exponential Smoothing. 
    Univariate time series without trend and seasonal components.
    '''
    model = SimpleExpSmoothing(data[key])
    model_fit = model.fit()
    return model_fit.predict(len(data), len(data))

def next_step_HWES(data, key='algo_price'):
    '''
    Holt Winters Exponential Smoothing.
    Univariate time series with trend and/or seasonal components.
    '''
    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    return model_fit.predict(len(data), len(data))

#https://medium.datadriveninvestor.com/time-series-prediction-using-sarimax-a6604f258c56
#https://medium.datadriveninvestor.com/time-series-prediction-using-sarimax-a6604f258c56
#https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
def next_step_SARIMAX(data, key='algo_price'):
    '''
    Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors.
    Univariate time series with trend and/or seasonal components and exogenous variables.
    '''
    model = SARIMAX(data[key], exog=data['algo_volume'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    #exog = ....
    return model_fit.predict(len(data), len(data), exog=[exog])
