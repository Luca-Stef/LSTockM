from utils import adjust_positions, 

if not trading_client.get_clock().is_open: print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Market closed, sleeping.')

adjust_positions()

print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Done for the day, sleeping until tomorrow')