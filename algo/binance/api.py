import requests
import datetime
import datetime as dt
import asyncio
import websockets
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

client = Client(api_key, api_secret)

# get market depth
depth = client.get_order_book(symbol='BNBBTC')

symbol = "BTCUSDT"

# url = f'https://api.binance.com/api/v3/depth?symbol={symbol}'
ws_url = 'wss://stream.binance.com:9443'

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_5MINUTE, "1 day ago UTC")


# socket manager using threads
twm = ThreadedWebsocketManager()
twm.start()

# depth cache manager using threads
dcm = ThreadedDepthCacheManager()
dcm.start()


def handle_socket_message(msg):
    print(f"message type: {msg['e']}")
    print(msg)


def handle_dcm_message(depth_cache):
    print(f"symbol {depth_cache.symbol}")
    print("top 5 bids")
    print(depth_cache.get_bids()[:5])
    print("top 5 asks")
    print(depth_cache.get_asks()[:5])
    print("last update time {}".format(depth_cache.update_time))


dcm.start_depth_cache(callback=handle_dcm_message, symbol='ETHBTC')
