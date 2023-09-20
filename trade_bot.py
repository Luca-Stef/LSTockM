from trade_utils import * 

if trading_client.get_clock().is_open:

    adjust_positions()

else:

    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Market closed.')

print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Done for the day, sleeping until tomorrow.')