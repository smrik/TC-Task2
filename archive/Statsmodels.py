# %%
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import Data as Data 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# %% data2 HU
data2 = pd.read_csv('data/dataset_task2_shortHU.csv')

data2['date'] = pd.to_datetime(data2['date'], format='%Y-%m-%d %H:%M:%S')
data2['hour'] = data2['date'].dt.hour
data2['minute'] = data2['date'].dt.minute
data2['DayOfWeek'] = data2['date'].dt.dayofweek
data2['DayOfMonth'] = data2['date'].dt.day

data2['shift_1d_IDA1'] = data2['IDA1price'].shift(96).dropna()
data2['shift_1w_IDA1'] = data2['IDA1price'].shift(96*7).dropna()
data2['shift_HU'] = data2['HUprice'].shift(96).dropna()

data2['shift_1d'] = data2['IDA2price'].shift(96).dropna()
data2['shift_2d'] = data2['IDA2price'].shift(96*2).dropna()
data2['shift_3d'] = data2['IDA2price'].shift(96*3).dropna()
data2['shift_8d'] = data2['IDA2price'].shift(96*8).dropna()
data2['shift_1w'] = data2['IDA2price'].shift(96*7).dropna()

holidays_map = {holiday: i for i, holiday in enumerate(data2['holiday'].unique())}
data2['holiday'] = data2['holiday'].map(holidays_map)

data2 = data2.dropna()
data2.sort_index(inplace=True)

data2.index = data2['date']
data2.drop(columns='date', inplace=True)
data2 = data2.iloc[:-24*4*1, :] # za pred
data2

# %% Data 2
X = data2[['hour', 'minute', 'IDA1price', 'volumes', 'forecast_production', 'SIPXprice', 'da_load',
		  'shift_1d', 'forecast_consumption', 'shift_2d', 'shift_8d', 'HUprice',
		  'DayOfWeek', 'holiday', 'shift_1d_IDA1']]
y = data2['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0, shuffle=False)

# Final prediction
split_date = '2024-11-01'  # Change this to your desired date if needed
# split_date = '2024-11-01'  # Change this to your desired date if needed
X_train2 = X[data2.index < split_date]
y_train2 = y[data2.index < split_date]
X_test2 = X[data2.index >= split_date]
y_test2 = y[data2.index >= split_date]

from statsmodels.datasets.danish_data import load
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.ardl import ardl_select_order

# _ = (data2 - data2.mean()).plot()

sel_res = ardl_select_order(
    data2['IDA2price'], 3, data2[["IDA2price"]], 3, ic="aic", trend="c"
)
print(f"The optimal order is: {sel_res.model.ardl_order}")

res = sel_res.model.fit()
res.summary()

# %% Train the model  ------- 222222--------
model = sklearn.linear_model.RANSACRegressor(random_state=42)
model.fit(X_train2, y_train2)

# Make predictions
pred = pd.Series(model.predict(X_test2), index=y_test2.index)
merge = pd.concat([y_test2, pred], axis=1, ignore_index=True)
merge.rename(columns={0: 'IDA2price', 1: 'predicted_price_IDA2'}, inplace=True)

#Analyse results
rmse = root_mean_squared_error(merge['IDA2price'], merge['predicted_price_IDA2'])
mae = mean_absolute_error(merge['IDA2price'], merge['predicted_price_IDA2'])

print(f'Mean Absolute Error: {mae}, RMSE: {rmse}', end='\n\n')
merge.plot(figsize=(18, 8))

# %%
output = merge['predicted_price_IDA2'].copy()
output.index = output.index.strftime('%Y-%m-%dT%H:%M:%S')
# output.to_csv('data/forecasts/forecast_11-16.csv', header=False)
output