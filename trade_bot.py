from utils import *

from alpaca.trading.stream import TradingStream

trading_stream = TradingStream('api-key', 'secret-key', paper=True)
breakpoint()
async def update_handler(data):
    # trade updates will arrive in our async handler
    print(data)

# subscribe to trade updates and supply the handler as a parameter
trading_stream.subscribe_trade_updates(update_handler)

# start our websocket streaming
trading_stream.run()

while True:

    if not trading_client.get_clock().is_open: print("Market closed, sleeping."); sleep_until(hour=15)

    adjust_positions()

    print("Done for the day, sleeping until tomorrow"); sleep_until(hour=15)
    new_comp = get_comp(position.symbol)
    if set(new_comp) != set(current_comp): print(f"{position.symbol} composition has changed")