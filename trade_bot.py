from utils import *

SEC_KEY = 'R6cEuW4cGnVxi50ZHBIOFEj07Z1cxXMalMIAPkI0' 
PUB_KEY = 'PK46ASAR06YRWR9CLKHK'
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(key_id= PUB_KEY, secret_key=SEC_KEY, base_url=BASE_URL)

aok_comp = get_comp("aok")

input_data = yf.download(aok_comp).iloc[-20:].Close.to_numpy()

aok_model = tf.keras.models.load_model('models/aok')

aok_pred = aok_model.predict(X.reshape(1,*X.shape))[0,0]
aok_curr = yf.download("aok").Close[-1]

breakpoint()

long = 0
short = 0

while True:
    # unload short position and buy or hold long position
    if aok_pred > aok_curr:
        if not long:
            api.submit_order('aok', qty=1, side='buy')
            long = 1
            short = 0

    # unload long position and short or hold short position
    else: 
        if not short:
            api.submit_order('aok', qty=1, side='sell')
            short = 1
            long = 0

    time.sleep(86400)
