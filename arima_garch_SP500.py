
# arima_garch_SP500.py

# codice python relativo all'articolo presente su tradingquant.it
# https://tradingquant.it/strategia-di-trading-sullindice-sp500-con-i-modelli-arima-e-garch/

import yfinance as yf
import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings("ignore")

symbol='^GSPC'
start = '2002-01-01'
end = '2018-12-31'
SP500 = yf.download(symbol, start=start, end=end)

log_ret = np.log(SP500['Adj Close']) - np.log(SP500['Adj Close'].shift(1))
log_ret.dropna(inplace=True)
# Creazione del dataset
windowLength = 500
foreLength = len(log_ret) - windowLength

windowed_ds = []
for i in range(foreLength-1):
    windowed_ds.append(log_ret[i:i + windowLength])

# creazione del dataframe forecasts con zeri
forecasts = log_ret.iloc[windowLength:].copy() * 0

"""
import pmdarima
import arch
def fit_arima(series, range_p=range(0, 6), range_q=range(0, 6)):
    final_order = (0, 0, 0)
    best_aic = np.inf
    arima = pmdarima.ARIMA(order=final_order)

    for p in range_p:
        for q in range_q:
            if (p == 0) and (q == 0):
                next
            arima.order = (p, 0, q)
            arima.fit(series)

            aic = arima.aic()

            if aic < best_aic:
                best_aic = aic
                final_order = (p, 0, q)

    arima.order = final_order
    return arima.fit(series)


for i, window in enumerate(windowed_ds):
    try:
        # ARIMA model
        arima = fit_arima(window)
        arima_pred = arima.predict(n_periods=1)

        # GARCH model
        garch = arch.arch_model(arima.resid())
        garch_fit = garch.fit(disp='off', show_warning=False, )
        garch_pred = garch_fit.forecast(horizon=1).mean.iloc[-1]['h.1']

        forecasts.iloc[i] = arima_pred + garch_pred

        print(f'Date {str(forecasts.index[i].date())} : Fitted ARIMA order {arima.order} - Prediction={forecasts.iloc[i]}')
    except:
        forecasts.iloc[i] = 0


# Memorizzazione dei nuovi segnali calcolati
forecasts.to_csv('prova.csv')
"""
forecasts = pd.read_csv('new_python_forecasts.csv')
forecasts.columns=['Date','Signal']
forecasts.set_index('Date', inplace=True)
# Otteniamo il periodo che ci interessa
forecasts = forecasts[(forecasts.index>='2004-01-01') & (forecasts.index<='2018-12-31')]

# Calcolo direzione delle previsioni
forecasts['Signal'] = np.sign(forecasts['Signal'])

forecasts.index = pd.to_datetime(forecasts.index)

# Creo un dataframe che contiene le statistiche della strategia
stats=SP500.copy()
stats['LogRets']=log_ret
stats = stats[(stats.index>='2004-01-01') & (stats.index<='2018-12-31')]
stats.loc[stats.index, 'StratSignal'] = forecasts.loc[stats.index, 'Signal']
stats['StratLogRets'] = stats['LogRets'] * stats['StratSignal']
stats.loc[stats.index, 'CumStratLogRets'] = stats['StratLogRets'].cumsum()
stats.loc[stats.index, 'CumStratRets'] = np.exp(stats['CumStratLogRets'])

# Calcolo del confronto con il benchmark
start_stats = pd.to_datetime('2004-01-02')
end_stats = pd.to_datetime('2012-12-31')

results = stats.loc[(stats.index > start_stats) & (stats.index < end_stats),
                    ['Adj Close', 'LogRets', 'StratLogRets']].copy()

results['CumLogRets'] = results['LogRets'].cumsum()
results['CumRets'] = 100 * (np.exp(results['CumLogRets']) - 1)

results['CumStratLogRets'] = results['StratLogRets'].cumsum()
results['CumStratRets'] = 100 * (np.exp(results['CumStratLogRets']) - 1)

buy_hold_first = SP500.loc[start_stats, 'Adj Close']
buy_hold_last = SP500.loc[end_stats, 'Adj Close']
buy_hold = (buy_hold_last-buy_hold_first)/buy_hold_first

strategy = np.exp(results.loc[results.index[-1], 'CumStratLogRets']) - 1

pct_pos_returns = (results['LogRets'] > 0).mean() * 100
pct_strat_pos_returns = (results['StratLogRets'] > 0).mean() * 100

print(f'Returns:')
print(f'Buy_n_Hold - Return in period: {100 * buy_hold:.2f}% - Positive returns: {pct_pos_returns:.2f}%')
print(f'Strategy - Return in period: {100 * strategy:.2f}% - Positive returns: {pct_strat_pos_returns:.2f}%')

import matplotlib.pyplot as plt

columns = ['CumLogRets', 'CumStratLogRets']
plot_df=results[columns]
plot_df.plot(figsize=(15,7))

print("")