import datetime
import time
import numpy as np
from scrts import *
import yfinance as yf
import requests
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pickle import dump, load
import smtplib
import pytz
from sklearn.preprocessing import RobustScaler
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass

trading_client = TradingClient(PUB_KEY, SEC_KEY, paper=True)

def get_comp(ticker, url="https://www.zacks.com/funds/etf/{}/holding"):

    with requests.Session() as req:
        req.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"})
        r = req.get(url.format(ticker))
        goal = re.findall(r'etf\\\/(.*?)\\', r.text)
        return goal

def adjust_positions():

    # Get all positions
    positions = trading_client.get_all_positions()

    if not positions: initialise_positions()

    for position in positions:

        # acquire a long position if price will rise
        if get_signal(position) == "long":
            if position.side.value == "short":
                close_order = trading_client.close_position(symbol_or_asset_id=position.symbol)
                wait_for_order()
                market_order = trading_client.submit_order(order_data=MarketOrderRequest(symbol=position.symbol, qty=abs(float(position.qty)), side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Acquired long position for {position.symbol}')
                return
            else:
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Holding long position for {position.symbol}')
                return

        # acquire a short position if price will drop
        else: 
            if position.side.value == "long": 
                close_order = trading_client.close_position(symbol_or_asset_id=position.symbol)
                wait_for_order()
                market_order = trading_client.submit_order(order_data=MarketOrderRequest(symbol=position.symbol, qty=abs(float(position.qty)), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Acquired short position for {position.symbol}')
                return
            else:
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Holding short position for {position.symbol}')
                return

def get_signal(position):

    # Get all models and comps here, and check if funds have changed etc
    scaler = load(open(f'models/{position.symbol.lower()}_scaler.pkl', 'rb'))
    model = tf.keras.models.load_model(f'models/{position.symbol.lower()}')
    current_comp = get_comp(position.symbol)
    
    # Get new forecast
    input_data = np.nan_to_num(yf.download(current_comp, progress=False).iloc[-20:][["Open", "High", "Low", "Close"]].to_numpy())
    input_data = scaler.transform(input_data)
    prediction = np.round(model.predict(input_data.reshape(1,*input_data.shape))[0,0])

    return "short" if prediction else "long"

def wait_for_order():
    for _ in range(5): # wait for order to fill
        time.sleep(5)
        if not trading_client.get_orders(): break


def initialise_positions():
    pass

def notify(message, receiver_email = 'lucastu2013@gmail.com'):

    # Email configuration
    sender_email = 'lucastu2013@gmail.com'
    sender_password = 'pbfonhdbrsrrqqvp'  # Use an App Password if 2-factor authentication is enabled
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