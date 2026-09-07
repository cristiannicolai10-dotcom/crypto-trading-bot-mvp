from pybit.unified_trading import HTTP
import datetime


session = HTTP(
    testnet=True
)


response = session.get_tickers(
    category="linear",
    symbol="BTCUSDT"
)


print(response)
