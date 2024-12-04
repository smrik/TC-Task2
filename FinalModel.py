# %% Imports
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import sklearn.linear_model
from sklearn.model_selection import TimeSeriesSplit
import Data as Data 
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

# %% loading the data
data = pd.read_csv('data/dataset_task2.csv')

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute
data['DayOfWeek'] = data['date'].dt.dayofweek
data['season'] = data['date'].dt.month%12 // 3 + 1

data['shift_1d_IDA1'] = data['IDA1price'].shift(96).dropna()

data['shift_1d'] = data['IDA2price'].shift(96).dropna()
data['shift_2d'] = data['IDA2price'].shift(96*2).dropna()
data['shift_1w'] = data['IDA2price'].shift(96*7).dropna()

# change holidays to numbers
holidays_map = {holiday: i for i, holiday in enumerate(data['holiday'].unique())}
data['holiday'] = data['holiday'].map(holidays_map)

data = data.dropna()
data.sort_index(inplace=True)

# drop data before 2023-01-01, tested to be better
data = data[data['date'] >= '2023-01-01']

data.index = data['date']
data.drop(columns='date', inplace=True)
print(data.columns)
data

# %% Removing outliers
from scipy import stats

# Print the old shape of the DataFrame
print("Old Shape: ", data.shape)

threshold_z = 2
z = np.abs(stats.zscore(data['IDA2price']))

outlier_indices = np.where(z > threshold_z)[0]
data.drop(data.index[outlier_indices], inplace=True)

# Print the new shape of the DataFrame
print("New Shape: ", data.shape)

# outlier check
fig, ax = plt.subplots()
fig.set_size_inches(18, 8)
ax.hist(data['IDA2price'], bins=20)

# %% Time Series Cross Validation - RANSAC () with cross_val_score
# used to cross validate the model
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv = TimeSeriesSplit(n_splits=100, test_size=96)

# Initialize an empty DataFrame to store the metrics
metrics_df = pd.DataFrame(columns=['Split', 'RMSE', 'MAPE', 'MAE'])

model = sklearn.linear_model.RANSACRegressor(random_state=42)

X = data[['forecast_consumption', 'holiday', 'T', 'cloud_cover', 'da_load', 'solar', 'IDA1price', 'HUprice', 'season', 'AT1price', 'AT2price',
			'price', 'volumes', 'hour', 'minute', 'DayOfWeek', 'shift_1d_IDA1', 'shift_1d', 'shift_2d', 'shift_1w']]
y = data['IDA2price']

# Cross Validation
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', verbose=0)
print("Cross Validation Scores: ", scores)
print("Cross Validation Score Mean: ", scores.mean())
print("Cross Validation Score Std: ", scores.std())

# %% Data
X = data[['forecast_consumption', 'holiday', 'T', 'cloud_cover', 'da_load', 'solar', 'IDA1price', 'HUprice', 'season', 'AT1price',
			'price', 'volumes', 'hour', 'minute', 'DayOfWeek', 'shift_1d_IDA1', 'shift_1d', 'shift_2d', 'shift_1w']]
y = data['IDA2price']

# Final prediction
split_date = '2024-11-27'
X_train = X[data.index < split_date]
y_train = y[data.index < split_date]
X_test = X[data.index >= split_date]
y_test = y[data.index >= split_date]

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

# %% output to file
output = merge['predicted_price_IDA2'].copy()
output.index = output.index.strftime('%Y-%m-%dT%H:%M:%S')
output.to_csv('data/forecasts/forecast_11-27.csv', header=False)
output
