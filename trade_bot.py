from utils import *

while True:

    if not trading_client.get_clock().is_open: sleep_until(hour=15)

    adjust_positions()

    print("Done for the day, sleeping until tomorrow"); sleep_until('15:30:00')
    new_comp = get_comp(position.symbol)
    if set(new_comp) != set(current_comp): print(f"{position.symbol} composition has changed")