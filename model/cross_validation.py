import datetime
import numpy as np
import pandas as pd
import sklearn
import pandas_datareader as pdr
import pylab as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    Si crea un DataFrame pandas che memorizza i rendimenti percentuali dei
    prezzi di chiusura rettificata di un titolo azionario ottenuta da Yahoo
    Finance, insieme a una serie di rendimenti ritardati dai giorni di negoziazione
    precedenti (i valori predefiniti ritardano di 5 giorni).
    Sono inclusi anche il volume degli scambi, così come la direzione del giorno precedente.
    """

    # Ottieni informazioni sulle azioni da Yahoo Finance
    ts = pdr.get_data_yahoo(symbol, start_date-datetime.timedelta(days=365), end_date)

    # Crea un nuovo Dataframe per i ritardi
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    # Crea la serie traslata dei ritardi dei prezzi di chiusura del periodo (giorno) precedente
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Crea il DataFrame dei ritorni
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # Se uno qualsiasi dei valori dei ritorni percentuali è uguale a zero, si impostano
    # a un numero piccolo (per non avere problemi con il modello QDA in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Crea la serie dei ritorni precedenti percentuali
    for i in range(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Crea la serie "Direction" (+1 o -1) che indica un giorno up/down
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret


def validation_set_poly(random_seeds, degrees, X, y):
    """
    Utilizza il metodo train_test_split per creare un set
    di addestramento e un set di convalida (50% per ciascuno)
    utilizzando separati campionamenti casuali "random_seeds"
    per modelli di regressione lineare di varia flessibilità
    """
    sample_dict = dict([("seed_%s" % i,[]) for i in range(1, random_seeds+1)])
    # Esegui un ciclo su ogni suddivisione casuale in una suddivisione train-test
    for i in range(1, random_seeds+1):
        print("Random: %s" % i)
        # Aumenta il grado di ordine polinomiale della regressione lineare
        for d in range(1, degrees+1):
            print("Degree: %s" % d)
            # Crea il modello, divide gli insiemi e li addestra
            polynomial_features = PolynomialFeatures(
                degree=d, include_bias=False
            )
            linear_regression = LinearRegression()
            model = Pipeline([
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression)
            ])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=i
            )
            model.fit(X_train, y_train)
            # Calcola il test MSE e lo aggiunge al
            # dizionario di tutte le curve di test
            y_pred = model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)
            sample_dict["seed_%s" % i].append(test_mse)
        # Converte queste liste in array numpy per calcolare la media
        sample_dict["seed_%s" % i] = np.array(sample_dict["seed_%s" % i])
    # Crea la serie delle "medie dei test MSE" colcolando la media
    # del test MSE per ogni grado dei modelli di regressione lineare,
    # attraverso tutti i campionamenti casuali
    sample_dict["avg"] = np.zeros(degrees)
    for i in range(1, random_seeds+1):
        sample_dict["avg"] += sample_dict["seed_%s" % i]
    sample_dict["avg"] /= float(random_seeds)
    return sample_dict


def plot_test_error_curves_vs(sample_dict, random_seeds, degrees):
    fig, ax = plt.subplots()
    ds = range(1, degrees+1)
    for i in range(1, random_seeds+1):
        ax.plot(ds, sample_dict["seed_%s" % i], lw=2, label='Test MSE - Sample %s' % i)
    ax.plot(ds, sample_dict["avg"], linestyle='--', color="black", lw=3, label='Avg Test MSE')
    ax.legend(loc=0)
    ax.set_xlabel('Degree of Polynomial Fit')
    ax.set_ylabel('Mean Squared Error')
    ax.set_ylim([0.0, 4.0])
    fig.set_facecolor('white')
    plt.show()


def k_fold_cross_val_poly(folds, degrees, X, y):
    n = len(X)
    kf = KFold(n, n_folds=folds)
    kf_dict = dict([("fold_%s" % i,[]) for i in range(1, folds+1)])
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        print("Fold: %s" % fold)
        X_train, X_test = X.ix[train_index], X.ix[test_index]
        y_train, y_test = y.ix[train_index], y.ix[test_index]
        # Aumenta il grado di ordine polinomiale della regressione lineare
        for d in range(1, degrees+1):
            print("Degree: %s" % d)
            # Crea il modello e lo addestra
            polynomial_features = PolynomialFeatures(
                degree=d, include_bias=False
            )
            linear_regression = LinearRegression()
            model = Pipeline([
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression)
            ])
            model.fit(X_train, y_train)
            # Calcola il test MSE e lo aggiunge al
            # dizionario di tutte le curve di test
            y_pred = model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)
            kf_dict["fold_%s" % fold].append(test_mse)
        # Converte queste liste in array numpy per calcolare la media
        kf_dict["fold_%s" % fold] = np.array(kf_dict["fold_%s" % fold])
    # Crea la serie dei "test MSE medi" calcolando la media dei
    # test MSE per ogni grado del modello di regressione lineare,
    # tramite ogni k folds.
    kf_dict["avg"] = np.zeros(degrees)
    for i in range(1, folds+1):
        kf_dict["avg"] += kf_dict["fold_%s" % i]
    kf_dict["avg"] /= float(folds)
    return kf_dict


def plot_test_error_curves_kf(kf_dict, folds, degrees):
    fig, ax = plt.subplots()
    ds = range(1, degrees+1)
    for i in range(1, folds+1):
        ax.plot(ds, kf_dict["fold_%s" % i], lw=2, label='Test MSE - Fold %s' % i)
    ax.plot(ds, kf_dict["avg"], linestyle='--', color="black", lw=3, label='Avg Test MSE')
    ax.legend(loc=0)
    ax.set_xlabel('Degree of Polynomial Fit')
    ax.set_ylabel('Mean Squared Error')
    ax.set_ylim([0.0, 4.0])
    fig.set_facecolor('white')
    plt.show()



if __name__ == "__main__":
    symbol = "^FTSE"
    symbol = "^GSPC"
    start_date = datetime.datetime(2004, 1, 1)
    end_date = datetime.datetime(2004, 12, 31)
    sp500_lags = create_lagged_series(symbol, start_date, end_date, lags=5)

    # Uso tutti e venti i ritorni di 2 giorni precedenti come
    # valori di predizione, con "Today" come risposta
    X = sp500_lags[[
        "Lag1", "Lag2", "Lag3", "Lag4", "Lag5",
        # "Lag6", "Lag7", "Lag8", "Lag9", "Lag10",
        # "Lag11", "Lag12", "Lag13", "Lag14", "Lag15",
        # "Lag16", "Lag17", "Lag18", "Lag19", "Lag20"
    ]]
    y = sp500_lags["Today"]
    degrees = 3

    # Visualizza le curve dell'errore di test per il set di validazione
    random_seeds = 10
    sample_dict_val = validation_set_poly(random_seeds, degrees, X, y)
    plot_test_error_curves_vs(sample_dict_val, random_seeds, degrees)

    # Visualizza le curve dell'errore di test per il set di k-fold CV
    folds = 10
    kf_dict = k_fold_cross_val_poly(folds, degrees, X, y)
    plot_test_error_curves_kf(kf_dict, folds, degrees)
