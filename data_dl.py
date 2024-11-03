import Data as Data
import pandas as pd
from io import BytesIO
import collections

dap = Data.getDayAheadPrices()
dap.to_csv('data/day_ahead_prices.csv', index=True)
