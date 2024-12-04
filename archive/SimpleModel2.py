# %%
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import Data as Data 
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

# %% data2 HU
data = pd.read_csv('data/dataset_task2_shortHU.csv')
# data = pd.read_csv('data/dataset_task2_shortHU_nointer.csv')

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

# drop data before 2023-01-01
data = data[data['date'] >= '2023-01-01']

data.index = data['date']
data.drop(columns='date', inplace=True)
# data = data.iloc[:-24*4*1, :] # za pred
print(data.columns)
data


# %% Data
X = data[['forecast_production', 'forecast_consumption', 'holiday', 'da_load',
       'solar', 'IDA1price', 'SIPXprice', 'HUprice', 'price', 'hour', 'minute',
         'shift_1d_IDA1', 'shift_1d', 'shift_1w']]
X = data[['forecast_consumption', 'holiday', 'T',
          'cloud_cover', 'da_load',
       'solar', 'IDA1price', 'HUprice', 'price',
       'volumes', 'hour', 'minute', 'DayOfWeek', 'shift_1d_IDA1', 'shift_1d',
       'shift_2d', 'shift_1w']]
y = data['IDA2price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0, shuffle=False)

# Final prediction
split_date = '2024-11-22'  # Change this to your desired date if needed
# split_date = '2024-11-01'  # Change this to your desired date if needed
X_train = X[data.index < split_date]
y_train = y[data.index < split_date]
X_test = X[data.index >= split_date]
y_test = y[data.index >= split_date]

# Train the model 
model = sklearn.linear_model.RANSACRegressor(random_state=42)
# model = sklearn.linear_model.RANSACRegressor()
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

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=96*4)

# Initialize an empty DataFrame to store the metrics
metrics_df = pd.DataFrame(columns=['Split', 'RMSE', 'MAPE', 'MAE'])

for train_index, test_index in tscv.split(data):
	X = data[['forecast_consumption', 'holiday', 'T', 'cloud_cover', 'da_load', 'solar', 'IDA1price', 'HUprice', 'seasonn', 'AT1price', 'AT2price',
			'price', 'volumes', 'hour', 'minute', 'DayOfWeek', 'shift_1d_IDA1', 'shift_1d', 'shift_2d', 'shift_1w']]
	# X = data[['forecast_consumption', 'T', 'da_load', 'IDA1price', 'HUprice', 'volumes',
	#           'price', 'hour', 'DayOfWeek', 'shift_1d_IDA1', 'shift_1d', 'shift_2d', 'shift_1w']]
	y = data['IDA2price']
	
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	
	# print(train_index, test_index)

	# Train the model 
	model = sklearn.linear_model.RANSACRegressor(random_state=42)
	# model = sklearn.linear_model.RANSACRegressor()
	model.fit(X_train, y_train)

	# Make predictions
	pred = pd.Series(model.predict(X_test), index=y_test.index)
	merge = pd.concat([y_test, pred], axis=1, ignore_index=True)
	merge.rename(columns={0: 'IDA2price', 1: 'predicted_price_IDA2'}, inplace=True)

	#Analyse results
	mape = mean_absolute_percentage_error(merge['IDA2price'], merge['predicted_price_IDA2'])
	rmse = root_mean_squared_error(merge['IDA2price'], merge['predicted_price_IDA2'])
	mae = mean_absolute_error(merge['IDA2price'], merge['predicted_price_IDA2'])

	print(f'Mean Absolute Error: {mae}, RMSE: {rmse}, MAPE: {mape*100:.2f}%', end='\n\n')
	merge.plot(figsize=(18, 8), title=f'Mean Absolute Error: {mae}, RMSE: {rmse}, MAPE: {mape*100:.2f}%')

	#Save the results
	metrics_df.loc[len(metrics_df)] = [str(train_index) + '-' + str(test_index), rmse, mape, mae]

	# perm_importance_result_train = permutation_importance(
	# model, X_train, y_train, n_repeats=50, scoring='neg_root_mean_squared_error'
	# )
	# plot_feature_importances(perm_importance_result_train, X_train.columns)



# %%
def plot_feature_importances(perm_importance_result, feat_name):
    """bar plot the feature importance"""

    fig, ax = plt.subplots()

    indices = perm_importance_result["importances_mean"].argsort()
    plt.barh(
        range(len(indices)),
        perm_importance_result["importances_mean"][indices],
        xerr=perm_importance_result["importances_std"][indices],
    )

    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])

perm_importance_result_train = permutation_importance(
    model, X_train, y_train, n_repeats=50, scoring='neg_root_mean_squared_error'
)

plot_feature_importances(perm_importance_result_train, X_train.columns)

# %%
model = sklearn.linear_model.RANSACRegressor(random_state=42)

from rfpimp import dropcol_importances

def custom_regression_r2_score(model, X, y, sample_weights):
    y_pred = model.predict(X)
    oob_score = root_mean_squared_error(y, y_pred)
    return oob_score

drop_importance = dropcol_importances(
    model, 
    X_train, 
    y_train, 
    None, 
    None, 
    custom_regression_r2_score)

drop_importance.sort_values(
    by=['Importance']).plot.barh(y='Importance')

# %%
output = merge['predicted_price_IDA2'].copy()
output.index = output.index.strftime('%Y-%m-%dT%H:%M:%S')
output.to_csv('data/forecasts/forecast_11-22.csv', header=False)
output

# %%
# %%
def plot_feature_importances(perm_importance_result, feat_name):
	"""bar plot the feature importance"""

	fig, ax = plt.subplots()

	indices = perm_importance_result["importances_mean"].argsort()
	plt.barh(
		range(len(indices)),
		perm_importance_result["importances_mean"][indices],
		xerr=perm_importance_result["importances_std"][indices],
	)

	ax.set_yticks(range(len(indices)))
	_ = ax.set_yticklabels(feat_name[indices])

perm_importance_result_train = permutation_importance(
	model, X_train, y_train, n_repeats=50, scoring='neg_root_mean_squared_error'
)

plot_feature_importances(perm_importance_result_train, X_train.columns)
