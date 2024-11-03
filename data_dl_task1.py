import Data as Data
import pandas as pd
from io import BytesIO
import collections

dap = Data.getDayAheadPrices()
dap.to_csv('../Task 1/datasets/main/day_ahead_prices.csv', index=True)

dav = Data.getDayAheadVolumes()
dav.to_csv('../Task 1/datasets/main/day_ahead_volumes.csv', index=True)
