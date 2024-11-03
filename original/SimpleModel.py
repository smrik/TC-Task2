import Data as Data 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#get historical IDA2(Intraday) prices on 15min level
IDA2_prices=Data.getIDA2Prices()

#get histrical DA (Day Ahead) prices on 15 min level (already resampled form 60 min level)
DA_prices=Data.getDayAheadPrices()

data = pd.merge(IDA2_prices, DA_prices, on='DeliveryDateTime', suffixes=('_IDA2', '_DA'))

# Drop rows with NaN values
data = data.dropna()

# Add hour and minute columns
data['Hour'] = data['DeliveryDateTime'].dt.hour
data['Minute'] = data['DeliveryDateTime'].dt.minute
data=data.set_index('DeliveryDateTime')

# Split date on train/test set
split_date = '2024-09-15'  # Change this to your desired date if needed
train_df = data[data.index < split_date]
test_df = data[data.index >= split_date]

X_train = train_df[['Hour', 'Minute','Price_DA']]
y_train = train_df['Price_IDA2']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prepare test features
X_test = test_df[['Hour', 'Minute','Price_DA']]

# Make predictions
test_df['Predicted_Price_IDA2'] = model.predict(X_test)


#Analyse results
mae = mean_absolute_error(test_df['Price_IDA2'], test_df['Predicted_Price_IDA2'])

print(f'Mean Absolute Error: {mae}')
test_df[['Predicted_Price_IDA2','Price_IDA2']].plot()