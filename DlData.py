# %%
import Data as Data
import pandas as pd

# %%
data = Data.getDayAheadPrices_1h()
data = data.sort_values(by='DeliveryDateTime', ascending=True).reset_index(drop=True)
data.tail()

# %%
data = data[data['DeliveryDateTime'] > "2022-01-01"]
data.tail()


# %%
data.to_csv('SIPX_hourly_data.csv', index=False)
# %%

data = Data.getDayAheadVolume()
data
# %%
