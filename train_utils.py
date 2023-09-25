import requests
import re
import sys
import os
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import pytz
import numpy as np
import datetime
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
from pickle import dump, load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce, AssetClass

def get_comp(ticker, url="https://www.zacks.com/funds/etf/{}/holding"):

    with requests.Session() as req:
        req.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"})
        r = req.get(url.format(ticker))
        goal = re.findall(r'etf\\\/(.*?)\\', r.text)
        return goal

def get_batch(fund_ticker):
    """
    Two approaches: weakest link approach is to discard entire ETF if one stock has no price data. exclusive approach is to just discard the stock with
    insufficient data. Maybe try discarding and report what percentage of fund was discarded, then decide if "insufficient data" threshold needs to be adjusted 
    or entire ETF should be discarded. Or just try zero padding so LSTM input can have variable dimension.
    """
    etf_comp = get_comp(ticker=fund_ticker)
    
    if not etf_comp: 
        print(f"Fund {fund_ticker} not found")
        return 0, 0
    
    print(f"Fetching {fund_ticker}")
    etf_price = yf.download(fund_ticker, progress=False)[["Open", "High", "Low", "Close"]].mean(axis=1)
    stock_prices = yf.download(etf_comp, progress=False)[["Open", "High", "Low", "Close"]]
    
    # Weakest link approach
    """max_window = np.array([stock_prices[c]["Close"].shape for c in etf_comp if stock_prices[c]["Close"].shape[0]]).min()
    print(f"Training with {max_window} days of data")
    X = np.array([stock_prices[c]["Close"].iloc[-max_window:] for c in etf_comp if stock_prices[c]["Close"].shape[0]]).T
    y = etf_price.iloc[-max_window:].to_numpy()"""
    # Padding approach
    max_window = min(etf_price.shape[0], stock_prices.shape[0])
    X = stock_prices.iloc[-max_window:].fillna(0).to_numpy()
    y = etf_price.iloc[-max_window:].fillna(0).to_numpy()
    # Stock discarding approach

    scaler = RobustScaler().fit(X)
    X = scaler.transform(X)

    return X, y, scaler

def create_windows(X, y, window_length=20, lookahead=10, shift=1, sample_rate=1):
    """
    Create windows of length window_length. shift is the interval between each window and sample rate is the interval 
    between each reading within a window. For example you can use a window of 60 readings but downsample by 2 so you 
    effectively get only 30 measurements in the same period of time. Stagger is the space between two windows if 
    choosing to use staggered pairs of windows. Lookahead is the gap between the target and the end of the training example time series.
    """
    Xs, ys = [], []    
    
    for i in range(0, len(X) - window_length - lookahead + 5, shift):

        Xs.append(X[i:i + window_length:sample_rate])
        ys.append(y[i + window_length - 5:i + window_length + lookahead - 5])
    
    ys = np.where(np.mean(np.array(ys)[:,:5], axis=1) < np.mean(np.array(ys)[:,5:], axis=1), 0, 1)
    return np.array(Xs), ys.reshape(*ys.shape, 1)

def LSTM_model(X_train, y_train):
    """
    Define LSTM model and topology. If you want to change from binary to multi-class classification you need to change the loss
    from binary crossentropy to categorical crossentropy, as well as the output shape and the activation of the final layer.
    """
    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=100, input_shape=[X_train.shape[1], X_train.shape[2]], return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=100, input_shape=[X_train.shape[1], X_train.shape[2]])))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dense(units=100, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])
    return model

def fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, name, epochs=20):
    """
    Fit training data for 20 epochs. For a final model please change this to at least 100 epochs to ensure it converges to maximum
    possible accuracy. Also does inference on test set, and uses the output and the ground truth to calculate accuracy and plot confusion.
    Also plots training and validation loss as a function of number of iterations.
    """
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title(f"Final loss {loss}")
    plt.savefig(f"figures/{name}_{loss}_loss.png")
    
    cm = confusion_matrix(y_test, np.round(y_pred))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    cm_display.plot()
    cm_display.ax_.set_title(f"Accuracy {accuracy}")
    plt.savefig(f"figures/{name}_{accuracy}_cm.png")
    print(f"Final accuracy is {accuracy}. \nTrained, evaulated {name} model and saved figures at figures/{name}_loss.png\n")

    return loss, accuracy