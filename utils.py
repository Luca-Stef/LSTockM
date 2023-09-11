import requests
import re
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

keys = ['XLU', 'XLRE']

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"}

def get_comp(ticker, url="https://www.zacks.com/funds/etf/{}/holding"):
    with requests.Session() as req:
        req.headers.update(headers)
        r = req.get(url.format(ticker))
        goal = re.findall(r'etf\\\/(.*?)\\', r.text)
        return goal

def get_batch(fund_ticker):

    etf_comp = get_comp(ticker=fund_ticker)

    print(f"Fetching {fund_ticker}")
    #etf_price = yf.download(fund_ticker, start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'))
    etf_price = yf.download(fund_ticker)
    
    stock_prices = {}
    for ticker in etf_comp:
        print(f"Fetching {ticker}")
        #stock_prices[ticker] = yf.download(f'{ticker}', start=start_date, end=end_date)
        stock_prices[ticker] = yf.download(f'{ticker}')
    
    max_window = np.array([stock_prices[c]["Close"].shape for c in etf_comp]).min()
    X = np.array([stock_prices[c]["Close"].iloc[-max_window:] for c in etf_comp]).T
    y = etf_price["Close"].iloc[-max_window:].to_numpy()
    return X, y

def create_windows(X, y, window_length=20, lookahead=5, shift=1, sample_rate=1, stagger=0):
    """
    Create windows of length window_length. shift is the interval between each window and sample rate is the interval 
    between each reading within a window. For example you can use a window of 60 readings but downsample by 2 so you 
    effectively get only 30 measurements in the same period of time. Stagger is the space between two windows if 
    choosing to use staggered pairs of windows. Lookahead is the gap between the target and the end of the training example time series.
    """
    if window_length == 1:
        return X, y
    
    Xs, ys = [], []
    if stagger:
        for i in range(0, len(X) - window_length - stagger, shift):
            Xs.append(np.r_[X[i:i + window_length:sample_rate],X[i + stagger:i + window_length + stagger:sample_rate]])
            ys.append(scipy.stats.mode(y[i: i + window_length + stagger])[0])
    
    elif not stagger:
        breakpoint()
        for i in range(0, len(X) - window_length, shift):
            Xs.append(X[i:(i + window_length):sample_rate])
            ys.append(scipy.stats.mode(y[i: i + window_length])[0])

    return np.array(Xs), np.array(ys).reshape(-1, 1)

def LSTM_model(X_train, y_train):
    """
    Define LSTM model and topology. If you want to change from binary to multi-class classification you need to change the loss
    from binary crossentropy to categorical crossentropy, as well as the output shape and the activation of the final layer.
    """
    model = keras.Sequential()
    #model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=5, input_shape=[X_train.shape[1], X_train.shape[2]])))
    model.add(keras.layers.LSTM(units=5, input_shape=[X_train.shape[1], X_train.shape[2]]))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=5, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    return model

def fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, name):
    """
    Fit training data for 20 epochs. For a final model please change this to at least 100 epochs to ensure it converges to maximum
    possible accuracy. Also does inference on test set, and uses the output and the ground truth to calculate accuracy and plot confusion.
    Also plots training and validation loss as a function of number of iterations.
    """

    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title(f"Final loss {loss}")
    plt.savefig(f"figures/{name}_loss.png")

    cm = confusion_matrix(y_test, np.round(y_pred))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    cm_display.plot()
    cm_display.ax_.set_title(f"Accuracy {accuracy}")
    plt.savefig(f"figures/{name}_cm.png")
    print(f"Trained, evaulated {name} model and created confusion matrix at figures/{name}_cm.png\n")