import Data as Data
import pandas as pd
from io import BytesIO
import collections

# southpool data
# Day Ahead Auction
# day ahead trading results 
dap = Data.getDayAheadPrices()
dap.to_csv('data/day_ahead_prices.csv', index=True)
dav = Data.getDayAheadVolumes()
dav.to_csv('data/day_ahead_volumes.csv', index=True)

# Day Ahead Auction
# market coupling results
# --- missing ---

# Intraday Auction
# IDA 1
ida1 = Data.getIDA1Prices()
ida1.to_csv('data/ida1_prices.csv', index=True)

# IDA 2
# 15 min time series - to forecast
ida2 = Data.getIDA2Prices()
ida2.to_csv('data/ida2_prices.csv', index=True)

# IDA 3 - too late

# Slovenain price index - SIPX
sipx = Data.getDayAheadPrices_15min()
sipx.to_csv('data/sipx_prices.csv', index=True)
