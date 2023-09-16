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
from pickle import dump
import tensorflow as tf
from tensorflow import keras
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce, AssetClass

SEC_KEY = 'R6cEuW4cGnVxi50ZHBIOFEj07Z1cxXMalMIAPkI0' 
PUB_KEY = 'PK46ASAR06YRWR9CLKHK'
trading_client = TradingClient(PUB_KEY, SEC_KEY, paper=True)

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
    etf_price = yf.download(fund_ticker, progress=False)["Close"]
    stock_prices = yf.download(etf_comp, progress=False)["Close"]
    
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
    dump(scaler, open(f'models/{fund_ticker}_scaler.pkl', 'wb'))

    return X, y

def create_windows(X, y, window_length=20, lookahead=2, shift=1, sample_rate=1):
    """
    Create windows of length window_length. shift is the interval between each window and sample rate is the interval 
    between each reading within a window. For example you can use a window of 60 readings but downsample by 2 so you 
    effectively get only 30 measurements in the same period of time. Stagger is the space between two windows if 
    choosing to use staggered pairs of windows. Lookahead is the gap between the target and the end of the training example time series.
    """
    Xs, ys = [], []    
    
    for i in range(0, len(X) - window_length - lookahead, shift):

        Xs.append(X[i:i + window_length:sample_rate])
        ys.append(y[i + window_length - 1:i + window_length + lookahead - 1])
    
    ys = np.where(np.array(ys)[:,0] < np.array(ys)[:,1], 0, 1)
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
    model.add(keras.layers.Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def fit_evaluate_LSTM(X_train, y_train, X_test, y_test, model, name, epochs=100):
    """
    Fit training data for 20 epochs. For a final model please change this to at least 100 epochs to ensure it converges to maximum
    possible accuracy. Also does inference on test set, and uses the output and the ground truth to calculate accuracy and plot confusion.
    Also plots training and validation loss as a function of number of iterations.
    """
    checkpoint_filepath = f'./models/tmp/{name}_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1, callbacks=[model_checkpoint_callback])
    model.load_weights(checkpoint_filepath)
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
    print(f"Trained, evaulated {name} model and saved figures at figures/{name}_loss.png\n")

def notify(message):

    # Email configuration
    sender_email = 'lucastu2013@gmail.com'
    sender_password = 'pbfonhdbrsrrqqvp'  # Use an App Password if 2-factor authentication is enabled
    receiver_email = 'lucastu2013@gmail.com'
    subject = "EC2 notification"

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # For Gmail
        server.starttls()  # Upgrade the connection to secure TLS
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print('Email sent successfully!')
    except Exception as e:
        print(f'Error: {str(e)}')

def sleep_until(hour=15, minute=0, second=0):
    
    # Define the UK time zone
    uk_timezone = pytz.timezone('Europe/London')

    # Get the current time in UK time
    current_time = datetime.datetime.now(uk_timezone)

    # Calculate the next 16:00 UK time
    next_16_oclock = current_time.replace(hour=hour, minute=minute, second=second, microsecond=0)

    # If the current time is after 16:00, add one day to get the next 16:00
    if current_time >= next_16_oclock:
        next_16_oclock += datetime.timedelta(days=1)

    # Calculate the time difference until the next 16:00
    time_difference = next_16_oclock - current_time

    # Convert the time difference to seconds
    time_difference_seconds = time_difference.total_seconds()

    # Sleep until the next 16:00 UK time
    time.sleep(time_difference_seconds)

def initialise_positions():
    pass

def adjust_positions():

    # Get all positions
    positions = trading_client.get_all_positions()

    if not positions: initialise_positions()

    for position in positions:

        # acquire a long position if price will rise
        if get_signal() == "long":
            if position.side.value == "short":
                close_order = trading_client.close_position(symbol_or_asset_id=position.symbol)
                time.sleep(10) # wait for order to fill
                market_order = trading_client.submit_order(order_data=MarketOrderRequest(symbol=position.symbol, qty=abs(float(position.qty)), side=OrderSide.BUY, time_in_force=TimeInForce.DAY))

        # acquire a short position if price will drop
        else: 
            if position.side.value == "long": 
                close_order = trading_client.close_position(symbol_or_asset_id=position.symbol)
                time.sleep(10)
                market_order = trading_client.submit_order(order_data=MarketOrderRequest(symbol=position.symbol, qty=abs(float(position.qty)), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))

def get_signal():

    # Get all models and comps here, and check if funds have changed etc
    model = tf.keras.models.load_model(f'models/{position.symbol.lower()}')
    current_comp = get_comp(position.symbol)

    # Get new forecast
    input_data = np.nan_to_num(yf.download(current_comp, progress=False).iloc[-20:].Close.to_numpy())
    price_pred = model.predict(input_data.reshape(1,*input_data.shape))[0,0]
    price_curr = yf.download(position.symbol, progress=False).Close[-1]

    max_idx = 0
    min_idx = 2

    if forecast[max_idx] > forecast[-1]:
        pass

    if forecast[min_idx] < forecast[-1]:
        pass

    if price_pred > price_curr: return "long"
    else: return "short"