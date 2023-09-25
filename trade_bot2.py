from urllib.request import urlopen
import certifi
import json

KEY = "86bd68e097af4c677e33f18b31651fc6"

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

symbol_list = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey={KEY}")
print(symbol_list[0])
breakpoint()
income_statement = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey={KEY}")
balance_sheet = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/income-statement-as-reported/AAPL?apikey={KEY}")
cash_flow =  get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey={KEY}")