import ccxt

from algo.bybit.utils import get_key, get_pub_key

# import ccxt.async_support as ccxt

username = get_pub_key()
secret = get_key()

bybit = ccxt.bybit({
    'apiKey': username,
    'secret': secret,
})

bybit_markets = bybit.load_markets()

# print(bybit.id, bybit_markets)

ret = bybit.fetch_balance({'coin': 'USDT', 'type': 'spot'})

# ['info']['result']
print(ret)

# sell one ฿ for market price and receive $ right now
# print(exmo.id, exmo.create_market_sell_order('BTC/USD', 1))

# limit buy BTC/EUR, you pay €2500 and receive ฿1  when the order is closed
# print(exmo.id, exmo.create_limit_buy_order('BTC/EUR', 1, 2500.00))

# pass/redefine custom exchange-specific order params: type, amount, price, flags, etc...
# kraken.create_market_buy_order('BTC/USD', 1, {'trading_agreement': 'agree'})
