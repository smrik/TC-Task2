import Data as Data
import pandas as pd
from io import BytesIO
import collections
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import time
from selenium.webdriver.common.keys import Keys
import openmeteo_requests
import requests_cache
from retry_requests import retry

# southpool data
# Day Ahead Auction - day ahead trading results 
dap = Data.getDayAheadPrices()
dap.to_csv('data/da/day_ahead_prices.csv', index=True)
dav = Data.getDayAheadVolumes()
dav.to_csv('data/da/day_ahead_volumes.csv', index=True)

#
# IDA 1 - for the last three years
#
# ida1_old = Data.getIDA1_Old()
# ida1_old['DeliveryDateTime'] = ida1_old.index
# ida1_old.to_csv('data/ida1_old_prices.csv', index=False)
ida1_old = pd.read_csv('data/ida/ida1_old_prices.csv')
ida1 = Data.getIDA1Prices()
ida1 = pd.concat([ida1_old, ida1])
ida1.to_csv('data/ida/ida1_prices.csv', index=True)

# 
# IDA 2 - for the last three years
#
# 15 min time series - to forecast
# ida2_old = Data.getIDA2_Old()
# ida2_old['DeliveryDateTime'] = ida2_old.index
# ida2_old.to_csv('data/ida2_old_prices.csv', index=False)
ida2_old = pd.read_csv('data/ida/ida2_old_prices.csv')
ida2 = Data.getIDA2Prices()
ida2 = pd.concat([ida2_old, ida2])
ida2.to_csv('data/ida/ida2_prices.csv', index=True)

# Slovenain price index - SIPX
sipx = Data.getDayAheadPrices_15min()
sipx.to_csv('data/other/sipx_prices.csv', index=True)

# 
# For downloading the ELES data, because their own csv had wrong demand forecast in the data
#
def eles_update(current_date, end_date, df_current):
	driver = webdriver.Chrome()
	driver.get("https://www.eles.si/prevzem-in-proizvodnja")

	datelist = pd.date_range(start = current_date + timedelta(hours=1),
						  	 end = end_date, inclusive='both', freq='d').tolist()

	for date in datelist:
		date_input = date.strftime("%d.%m.%Y")
		date_input_box = driver.find_element(By.CLASS_NAME, "datepicker")
		date_input_box.clear()
		date_input_box.send_keys(date_input)
		date_input_box.send_keys(Keys.TAB)
		pokazi_tabelo = driver.find_element(By.ID, "dnn_ctr1215_View_ctl00_btn_ShowTimedDataTabela")
		pokazi_tabelo.click()
		time.sleep(1)
		table = driver.find_element(By.ID, "dnn_ctr1215_View_ctl00_RadTabbedDataL_ctl00")
		df_table = pd.read_html(table.get_attribute('outerHTML'))
		df_table[0].index = pd.date_range(start=date, periods=24, freq='h')
		df_current = pd.concat([df_current, df_table[0]])

	df_current.drop(columns=["Unnamed: 0"], inplace=True)
	df_current.to_csv("data/other/ELES.csv")

df_current = pd.read_csv("data/other/ELES.csv", index_col=0)


start_date = datetime(2022, 1, 1)
current_date = datetime.fromisoformat(df_current.index[-1])
end_date = datetime.today() + timedelta(days=1)
end_date = datetime.replace(end_date, hour=23, minute=0, second=0, microsecond=0)

if current_date < end_date:
	eles_update(current_date, end_date, df_current)
else:
	print("Data is up to date")


#
# For Dowloading the weather forecasts for the next days
#

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
	"latitude": 46.2237,
	"longitude": 14.4576,
	"hourly": ["temperature_2m", "temperature_2m_previous_day1", "temperature_2m_previous_day2", "temperature_2m_previous_day3", "temperature_2m_previous_day4", "temperature_2m_previous_day5", "temperature_2m_previous_day6", "temperature_2m_previous_day7", "cloud_cover", "cloud_cover_previous_day1", "cloud_cover_previous_day2", "cloud_cover_previous_day3", "cloud_cover_previous_day4", "cloud_cover_previous_day5", "cloud_cover_previous_day6", "cloud_cover_previous_day7", "direct_normal_irradiance", "direct_normal_irradiance_previous_day1", "direct_normal_irradiance_previous_day2", "direct_normal_irradiance_previous_day3", "direct_normal_irradiance_previous_day4", "direct_normal_irradiance_previous_day5", "direct_normal_irradiance_previous_day6", "direct_normal_irradiance_previous_day7"],
	"timezone": "Europe/Berlin",
	"past_days": 14
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(8).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(16).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.to_csv("data/weather/weather_fore_11-27.csv", index=False)
print(hourly_dataframe)