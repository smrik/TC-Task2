# %%
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import Data as Data
import itertools
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

# %% data2 HU
data = pd.read_csv('data/dataset_task2_shortHU.csv')

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute
data['DayOfWeek'] = data['date'].dt.dayofweek
# data2['DayOfMonth'] = data2['date'].dt.day

data['shift_1d_IDA1'] = data['IDA1price'].shift(96).dropna()
# data2['shift_1w_IDA1'] = data2['IDA1price'].shift(96*7).dropna()
# data2['shift_HU'] = data2['HUprice'].shift(96).dropna()

data['shift_1d'] = data['IDA2price'].shift(96).dropna()
data['shift_2d'] = data['IDA2price'].shift(96*2).dropna()
data['shift_1w'] = data['IDA2price'].shift(96*7).dropna()

holidays_map = {holiday: i for i, holiday in enumerate(data['holiday'].unique())}
data['holiday'] = data['holiday'].map(holidays_map)

data = data.dropna()
data.sort_index(inplace=True)

data.index = data['date']
data.drop(columns='date', inplace=True)
# data = data.iloc[:-24*4*1, :] # za pred
print(data.columns)
data


# %% Data
X = data[['forecast_production', 'forecast_consumption', 'holiday', 'da_load',
       'solar', 'IDA1price', 'SIPXprice', 'HUprice', 'price', 'hour', 'minute',
         'shift_1d_IDA1', 'shift_1d', 'shift_1w']]
X = data[['forecast_consumption', 'shift_1w', 'cloud_cover', 'solar',
          'IDA1price', 'HUprice', 'price', 'hour', 'minute', 'shift_1d_IDA1', 'shift_1d']]
y = data['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0, shuffle=False)

# Final prediction
# split_date = '2024-11-18'  # Change this to your desired date if needed
# # split_date = '2024-10-01'  # Change this to your desired date if needed
# X_train = X[data.index < split_date]
# y_train = y[data.index < split_date]
# X_test = X[data.index >= split_date]
# y_test = y[data.index >= split_date]

# Train the model 
model = sklearn.linear_model.RANSACRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
pred = pd.Series(model.predict(X_test), index=y_test.index)
merge = pd.concat([y_test, pred], axis=1, ignore_index=True)
merge.rename(columns={0: 'IDA2price', 1: 'predicted_price_IDA2'}, inplace=True)

#Analyse results
rmse = root_mean_squared_error(merge['IDA2price'], merge['predicted_price_IDA2'])
mae = mean_absolute_error(merge['IDA2price'], merge['predicted_price_IDA2'])

print(f'Mean Absolute Error: {mae}, RMSE: {rmse}', end='\n\n')
merge.plot(figsize=(18, 8))

# %%

import time
from datetime import timedelta
start_time = time.time()
model = sklearn.linear_model.RANSACRegressor(random_state=42)

X2 = data[['forecast_consumption', 'holiday', 'T', 'shift_2d', 'shift_1w',
          'cloud_cover', 'da_load', 'solar', 'IDA1price', 'HUprice', 'price',
       'volumes', 'hour', 'minute', 'DayOfWeek', 'shift_1d_IDA1', 'shift_1d',]]
y2 = data['IDA2price']

# feat_perms = []
# for i in range(1, len(X2.columns)):
	# f_perm = list(itertools.combinations(X2.columns, i))
	# feat_perms.extend(f_perm)

feat_perms = list(itertools.combinations(X2.columns, 2))

results_dict = {}
for perm in feat_perms:
    X2_perm = data[list(perm)]
    X_train, X_test, y_train, y_test = train_test_split(X2_perm, y2, test_size=0.05, random_state=42, shuffle=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    results_dict[perm] = rmse

results_df = pd.DataFrame(list(results_dict.items()), columns=['Feature Combination', 'RMSE'])
results_df.sort_values(by='RMSE', inplace=True)
results_df

end_time = time.time()
print(f"Time taken: {timedelta(seconds=end_time - start_time)}")

# %%
from joblib import Parallel, delayed
import time
from datetime import timedelta
import itertools

def train_model(perm):
    X2_perm = data[list(perm)]
    X_train, X_test, y_train, y_test = train_test_split(X2_perm, y2, test_size=0.05, random_state=42, shuffle=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return perm, rmse

feat_perms = []
for r in range(1, len(X2.columns) + 1):
    feat_perms.extend(itertools.combinations(X2.columns, r))

start_time = time.time()
results = Parallel(n_jobs=-1)(delayed(train_model)(perm) for perm in feat_perms)
end_time = time.time()

results_dict = dict(results)
results_df = pd.DataFrame(list(results_dict.items()), columns=['Feature Combination', 'RMSE'])

print(f"Time taken: {timedelta(seconds=end_time - start_time)}")
print(results_df)

# %%
from joblib import Parallel, delayed
import time
from datetime import timedelta
import itertools
from tqdm import tqdm

def train_model(perm):
    X2_perm = data[list(perm)]
    X_train, X_test, y_train, y_test = train_test_split(X2_perm, y2, test_size=0.05, random_state=42, shuffle=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return perm, rmse

feat_perms = []
for r in range(1, len(X2.columns) + 1):
    feat_perms.extend(itertools.combinations(X2.columns, r))

start_time = time.time()
results = Parallel(n_jobs=-1, verbose=10)(delayed(train_model)(perm) for perm in tqdm(feat_perms, desc="Training models"))
end_time = time.time()

results_dict = dict(results)
results_df = pd.DataFrame(list(results_dict.items()), columns=['Feature Combination', 'RMSE'])

print(f"Time taken: {timedelta(seconds=end_time - start_time)}")
print(results_df)

# %%
output = merge['predicted_price_IDA2'].copy()
output.index = output.index.strftime('%Y-%m-%dT%H:%M:%S')
output.to_csv('data/forecasts/forecast_11-18.csv', header=False)
output
