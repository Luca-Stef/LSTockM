import pyEX

IEX_CLOUD_API_TOKEN = "pk_10e650123a8649a2857cf922305bfb3e"

symbol = 'AAPL'
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
data = requests.get(api_url).json()
print(data)