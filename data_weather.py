# %%

import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": [46.2237, 46.0833, 45.5469],
	"longitude": [14.4576, 15, 13.7294],
	"start_date": "2024-01-01",
	"end_date": "2024-11-17",
	"hourly": ["temperature_2m", "cloud_cover", "direct_normal_irradiance"],
	"timezone": "Europe/Berlin"
}
responses = openmeteo.weather_api(url, params=params)

merge = pd.DataFrame()

# Process first location. Add a for-loop for multiple locations or weather models
for response in responses:
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	lat_lon = "_{0:.2f}°N{1:.2f}°E".format(response.Latitude(), response.Longitude())

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
	hourly_direct_normal_irradiance = hourly.Variables(2).ValuesAsNumpy()

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
	merge = pd.merge(merge, hourly_dataframe, how = "outer",
				  left_index=True, right_index=True, suffixes = ("", lat_lon))

merge.to_csv("data/weather/weather24.csv", index=False)
print(hourly_dataframe)


# %%
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

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
hourly_temperature_2m_previous_day1 = hourly.Variables(1).ValuesAsNumpy()
hourly_temperature_2m_previous_day2 = hourly.Variables(2).ValuesAsNumpy()
hourly_temperature_2m_previous_day3 = hourly.Variables(3).ValuesAsNumpy()
hourly_temperature_2m_previous_day4 = hourly.Variables(4).ValuesAsNumpy()
hourly_temperature_2m_previous_day5 = hourly.Variables(5).ValuesAsNumpy()
hourly_temperature_2m_previous_day6 = hourly.Variables(6).ValuesAsNumpy()
hourly_temperature_2m_previous_day7 = hourly.Variables(7).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(8).ValuesAsNumpy()
hourly_cloud_cover_previous_day1 = hourly.Variables(9).ValuesAsNumpy()
hourly_cloud_cover_previous_day2 = hourly.Variables(10).ValuesAsNumpy()
hourly_cloud_cover_previous_day3 = hourly.Variables(11).ValuesAsNumpy()
hourly_cloud_cover_previous_day4 = hourly.Variables(12).ValuesAsNumpy()
hourly_cloud_cover_previous_day5 = hourly.Variables(13).ValuesAsNumpy()
hourly_cloud_cover_previous_day6 = hourly.Variables(14).ValuesAsNumpy()
hourly_cloud_cover_previous_day7 = hourly.Variables(15).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(16).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day1 = hourly.Variables(17).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day2 = hourly.Variables(18).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day3 = hourly.Variables(19).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day4 = hourly.Variables(20).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day5 = hourly.Variables(21).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day6 = hourly.Variables(22).ValuesAsNumpy()
hourly_direct_normal_irradiance_previous_day7 = hourly.Variables(23).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["temperature_2m_previous_day1"] = hourly_temperature_2m_previous_day1
hourly_data["temperature_2m_previous_day2"] = hourly_temperature_2m_previous_day2
hourly_data["temperature_2m_previous_day3"] = hourly_temperature_2m_previous_day3
hourly_data["temperature_2m_previous_day4"] = hourly_temperature_2m_previous_day4
hourly_data["temperature_2m_previous_day5"] = hourly_temperature_2m_previous_day5
hourly_data["temperature_2m_previous_day6"] = hourly_temperature_2m_previous_day6
hourly_data["temperature_2m_previous_day7"] = hourly_temperature_2m_previous_day7
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_previous_day1"] = hourly_cloud_cover_previous_day1
hourly_data["cloud_cover_previous_day2"] = hourly_cloud_cover_previous_day2
hourly_data["cloud_cover_previous_day3"] = hourly_cloud_cover_previous_day3
hourly_data["cloud_cover_previous_day4"] = hourly_cloud_cover_previous_day4
hourly_data["cloud_cover_previous_day5"] = hourly_cloud_cover_previous_day5
hourly_data["cloud_cover_previous_day6"] = hourly_cloud_cover_previous_day6
hourly_data["cloud_cover_previous_day7"] = hourly_cloud_cover_previous_day7
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
hourly_data["direct_normal_irradiance_previous_day1"] = hourly_direct_normal_irradiance_previous_day1
hourly_data["direct_normal_irradiance_previous_day2"] = hourly_direct_normal_irradiance_previous_day2
hourly_data["direct_normal_irradiance_previous_day3"] = hourly_direct_normal_irradiance_previous_day3
hourly_data["direct_normal_irradiance_previous_day4"] = hourly_direct_normal_irradiance_previous_day4
hourly_data["direct_normal_irradiance_previous_day5"] = hourly_direct_normal_irradiance_previous_day5
hourly_data["direct_normal_irradiance_previous_day6"] = hourly_direct_normal_irradiance_previous_day6
hourly_data["direct_normal_irradiance_previous_day7"] = hourly_direct_normal_irradiance_previous_day7

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe.to_csv("data/weather/weather_fore.csv", index=False)
print(hourly_dataframe)
# %%
