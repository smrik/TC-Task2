# %%
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import Data as Data 
import pandas as pd
import numpy as np
import xgboost as xg
import sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# %% data
data = pd.read_csv('data/dataset_task2.csv')

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute
data['DayOfWeek'] = data['date'].dt.dayofweek

data['shift_1d_IDA1'] = data['IDA1price'].shift(96).dropna()
data['shift_1w_IDA1'] = data['IDA1price'].shift(96*7).dropna()

data['shift_1d'] = data['IDA2price'].shift(96).dropna()
data['shift_2d'] = data['IDA2price'].shift(96*2).dropna()
data['shift_3d'] = data['IDA2price'].shift(96*3).dropna()
data['shift_8d'] = data['IDA2price'].shift(96*8).dropna()
data['shift_1w'] = data['IDA2price'].shift(96*7).dropna()

holidays_map = {holiday: i for i, holiday in enumerate(data['holiday'].unique())}
data['holiday'] = data['holiday'].map(holidays_map)

data = data.dropna()
data.sort_index(inplace=True)

data.index = data['date']
data.drop(columns='date', inplace=True)
# data = data.iloc[:-24*4*1, :] # za pred
data

# %% data2 HU
data2 = pd.read_csv('data/dataset_task2_shortHU.csv')

data2['date'] = pd.to_datetime(data2['date'], format='%Y-%m-%d %H:%M:%S')
data2['hour'] = data2['date'].dt.hour
data2['minute'] = data2['date'].dt.minute
data2['DayOfWeek'] = data2['date'].dt.dayofweek
# data2['DayOfMonth'] = data2['date'].dt.day

data2['shift_1d_IDA1'] = data2['IDA1price'].shift(96).dropna()
# data2['shift_1w_IDA1'] = data2['IDA1price'].shift(96*7).dropna()
# data2['shift_HU'] = data2['HUprice'].shift(96).dropna()

data2['shift_1d'] = data2['IDA2price'].shift(96).dropna()
data2['shift_2d'] = data2['IDA2price'].shift(96*2).dropna()
data2['shift_1w'] = data2['IDA2price'].shift(96*7).dropna()

holidays_map = {holiday: i for i, holiday in enumerate(data2['holiday'].unique())}
data2['holiday'] = data2['holiday'].map(holidays_map)

data2 = data2.dropna()
data2.sort_index(inplace=True)

data2.index = data2['date']
data2.drop(columns='date', inplace=True)
# data2 = data2.iloc[:-24*4*1, :] # za pred
print(data2.columns)
data2

# %% Data 1
X = data[['forecast_consumption', 'shift_1w', 'cloud_cover', 'solar', 'IDA1price', 'HUprice',
'price', 'hour', 'minute', 'shift_1d_IDA1', 'shift_1d']]
y = data['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0, shuffle=False)

# Final prediction
# split_date = '2024-11-12'  # Change this to your desired date if needed
split_date = '2024-11-01'  # Change this to your desired date if needed
X_train = X[data.index < split_date]
y_train = y[data.index < split_date]
X_test = X[data.index >= split_date]
y_test = y[data.index >= split_date]

# %% Data 2
X = data2[['forecast_consumption', 'shift_1w', 'cloud_cover', 'solar', 'IDA1price',
		   'HUprice', 'price', 'hour', 'minute', 'shift_1d_IDA1', 'shift_1d']]
y = data2['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0, shuffle=False)

# Final prediction
split_date = '2024-11-19'  # Change this to your desired date if needed
split_date = '2024-11-01'  # Change this to your desired date if needed
X_train2 = X[data2.index < split_date]
y_train2 = y[data2.index < split_date]
X_test2 = X[data2.index >= split_date]
y_test2 = y[data2.index >= split_date]

# %% Data 3
X = data2[['hour', 'minute', 'IDA1price', 'volumes', 'forecast_production', 'SIPXprice', 'da_load',
		  'shift_1d', 'forecast_consumption', 'shift_2d', 'shift_8d', 'HUprice',
		  'DayOfWeek', 'holiday', 'shift_1d_IDA1']]
y = data2['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0, shuffle=False)

# Final prediction
# split_date = '2024-11-12'  # Change this to your desired date if needed
split_date = '2024-11-01'  # Change this to your desired date if needed
X_train3 = X[data2.index < split_date]
y_train3 = y[data2.index < split_date]
X_test3 = X[data2.index >= split_date]
y_test3 = y[data2.index >= split_date]

# %% Train the model ------- 11111 --------
model = linear_model.RANSACRegressor()
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

#
#
#

# %% Train the model XGBoost  ------- 333333 --------
train_dmatrix = xg.DMatrix(data = X_train3, label = y_train3) 
test_dmatrix = xg.DMatrix(data = X_test3, label = y_test3)

param = {"booster":"dart", "objective":"reg:squarederror",
		 'learning_rate': 0.1, 'seed': 42, 'eval_metric': 'rmse'} 

xgb_r = xg.XGBRegressor(dtrain = train_dmatrix, num_boost_round = 100) 
xgb_r.fit(X_train3, y_train3)
# pred = xgb_r.predict(X_test3)
pred = pd.Series(xgb_r.predict(X_test3), index=y_test3.index)

# Make predictions
merge = pd.concat([y_test3, pred], axis=1, ignore_index=True)
merge.rename(columns={0: 'IDA2price', 1: 'predicted_price_IDA2'}, inplace=True)

#Analyse results
rmse = root_mean_squared_error(merge['IDA2price'], merge['predicted_price_IDA2'])
mae = mean_absolute_error(merge['IDA2price'], merge['predicted_price_IDA2'])

print(f'Mean Absolute Error: {mae}, RMSE: {rmse}', end='\n\n')
merge.plot(figsize=(18, 8))

#
#
#
# %% Grid search
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV

# Define the hyperparameters to tune
# param_grid = {
#     'learning_rate': [0.01, 0.1, 0.5],
#     'max_depth': [3, 5, 10],
#     'min_samples_leaf': [1, 5, 10]
# }

param_grid = {
	'learning_rate': (np.logspace(0.001, 1, 20)/1000).tolist(),
	'max_iter': [1000],
	'max_depth': [3],
	'min_samples_leaf': [1],
	'l2_regularization': [0.1, 0.05, 0.01, 0.005, 0.001]
}

# Best parameters: {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 1}
# Best model: HistGradientBoostingRegressor(learning_rate=0.05, max_depth=3, min_samples_leaf=1)
# RMSE: 37.825313081853125
# np.arange(11, 17, 0.5).tolist()
# param_grid = {
# 	'learning_rate': 0.0055,
# 	'max_iter': [1000],
# 	'max_depth': [3],
# 	'min_samples_leaf': [1],
# 	'l2_regularization': [0.1]
# }

# Define the GBR model
gbr = HistGradientBoostingRegressor(random_state=0)

# Perform grid search
grid_search = HalvingGridSearchCV(gbr,
								  param_grid,
								  cv=5,
								  scoring='neg_root_mean_squared_error',
								  n_jobs=-1,
								  verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_
print("Best model:", best_model)

# Evaluate the best model
forecast = best_model.predict(X_test)
mse = root_mean_squared_error(y_test, forecast)
print("RMSE:", mse)


# %%
output = merge['predicted_price_IDA2'].copy()
output.index = output.index.strftime('%Y-%m-%dT%H:%M:%S')
output.to_csv('data/forecasts/forecast_11-19.csv', header=False)
output