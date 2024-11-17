# %%
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import Data as Data 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# %%
data = pd.read_csv('data/dataset_task2.csv')

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute
data['DayOfWeek'] = data['date'].dt.dayofweek

data['shift_1d_IDA1'] = data['IDA1price'].shift(97).dropna()
data['shift_1w_IDA1'] = data['IDA1price'].shift(96*7).dropna()

data['shift_1d'] = data['IDA2price'].shift(96).dropna()
data['shift_1w'] = data['IDA2price'].shift(96*7).dropna()

holidays_map = {holiday: i for i, holiday in enumerate(data['holiday'].unique())}
data['holiday'] = data['holiday'].map(holidays_map)

data = data.dropna()

data.index = data['date']
data.drop(columns='date', inplace=True)
# data = data.iloc[:-24*4*2, :] # za pred
data

# %%
X = data[['hour', 'minute', 'IDA1price', 'volumes', 'forecast_production', 'SIPXprice', 'da_load',
		  'shift_1d', 'shift_1w', 'forecast_consumption', 'DayOfWeek', 'holiday', 'shift_1d_IDA1']]
y = data['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

# Final prediction
# split_date = '2024-11-12'  # Change this to your desired date if needed
# X_train = X[data.index < split_date]
# y_train = y[data.index < split_date]
# X_test = X[data.index >= split_date]
# y_test = y[data.index >= split_date]

# %%
# Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
# Mean Absolute Error: 29.680194037123943, RMSE: 45.76756707148307

model = HistGradientBoostingRegressor(loss='squared_error',
									  learning_rate=0.0055,
									  max_iter=1000,
									  max_depth=3,
									  min_samples_leaf=1,
									  l2_regularization=0.1,
									  random_state=42)
model.fit(X_train, y_train)
# Mean Absolute Error: 28.154867233189567, RMSE: 48.742037443778194

# model = GradientBoostingRegressor(random_state=0,
# 								learning_rate=0.1,
# 								n_estimators=200,
# 								max_depth=5, 
# 								min_samples_split=10)
# model.fit(X_train, y_train)
# Mean Absolute Error: 26.49973051900755, RMSE: 49.67686869135909

# Make predictions
pred = pd.Series(model.predict(X_test), index=y_test.index)
merge = pd.concat([y_test, pred], axis=1, ignore_index=True)
merge.rename(columns={0: 'IDA2price', 1: 'predicted_price_IDA2'}, inplace=True)

#Analyse results
rmse = root_mean_squared_error(merge['IDA2price'], merge['predicted_price_IDA2'])
mae = mean_absolute_error(merge['IDA2price'], merge['predicted_price_IDA2'])

print(f'Mean Absolute Error: {mae}, RMSE: {rmse}', end='\n\n')
merge.plot(figsize=(18, 8))

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
output.to_csv('data/forecasts/forecast_11-12.csv', header=False)
output
# %%
