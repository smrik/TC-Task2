# %%
import pandas as pd
import requests
from io import BytesIO
import collections

# %%
def getDayAheadPrices_15min():
	"""
	DayAhead Auction: Auction at 12:00 for hours in the next day

	Returns
	-------
	data : Pandas DataFrame with column DateTime(CET) and electricity price[€/MWh] 

	"""
	url = "https://www.bsp-southpool.com/sipx.html?file=files/documents/trading/SIPX_en.xlsx&cid=382"
	
	# Download the Excel file
	response = requests.get(url)
	response.raise_for_status()
	
	# Load the Excel file into a pandas DataFrame
	excel_file = BytesIO(response.content)
	data = pd.read_excel(excel_file, skiprows=1)  # Skip the header row
	
	# Select relevant columns: Delivery Date and SIPX1 to SIPX24
	columns = ['Delivery Date'] + [f'SIPX{i}' for i in range(1, 25)]
	data = data.iloc[:, [0] + list(range(3, 27))]
	
	# Rename columns
	data.columns = columns
	
	# Convert the Delivery Date to datetime
	data['Delivery Date'] = pd.to_datetime(data['Delivery Date'])

	data = pd.melt(data, id_vars=['Delivery Date'], value_vars=[f'SIPX{i}' for i in range(1, 25)],
					var_name='Hour', value_name='Price')
	data['Hour'] = data['Hour'].str.replace('SIPX', '').astype(int)-1
	data['DeliveryDateTime'] = pd.to_datetime(data['Delivery Date']) + pd.to_timedelta(data['Hour'], unit='H')
	
	# Localize to CET (Central European Time)
	data['DeliveryDateTime'] = data['DeliveryDateTime'].dt.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')
	data=data.dropna()

	# Resample to 15-minute frequency, using forward fill to maintain the hourly value
	intervals = pd.DataFrame({'Minutes': [0, 15, 30, 45]})
	# Perform a cross join
	data = data.merge(intervals, how='cross')
	# Add the intervals to the DeliveryDateTime
	data['DeliveryDateTime'] = data['DeliveryDateTime']  + pd.to_timedelta(data['Minutes'], unit='m')

	return data[['DeliveryDateTime', 'Price']]
# %%
def getDayAheadPrices_1h():
	"""
	DayAhead Auction: Auction at 12:00 for hours in the next day

	Returns
	-------
	data : Pandas DataFrame with column DateTime(CET) and electricity price[€/MWh] 

	"""
	url = "https://www.bsp-southpool.com/sipx.html?file=files/documents/trading/SIPX_en.xlsx&cid=382"
	
	# Download the Excel file
	response = requests.get(url)
	response.raise_for_status()
	
	# Load the Excel file into a pandas DataFrame
	excel_file = BytesIO(response.content)
	data = pd.read_excel(excel_file, skiprows=1)  # Skip the header row
	
	# Select relevant columns: Delivery Date and SIPX1 to SIPX24
	columns = ['Delivery Date'] + [f'SIPX{i}' for i in range(1, 25)]
	data = data.iloc[:, [0] + list(range(3, 27))]
	
	# Rename columns
	data.columns = columns
	
	# Convert the Delivery Date to datetime
	data['Delivery Date'] = pd.to_datetime(data['Delivery Date'])

	data = pd.melt(data, id_vars=['Delivery Date'], value_vars=[f'SIPX{i}' for i in range(1, 25)],
					var_name='Hour', value_name='Price')
	data['Hour'] = data['Hour'].str.replace('SIPX', '').astype(int)-1
	data['DeliveryDateTime'] = pd.to_datetime(data['Delivery Date']) + pd.to_timedelta(data['Hour'], unit='H')
	
	# Localize to CET (Central European Time)
	data['DeliveryDateTime'] = data['DeliveryDateTime'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT', )
	data=data.dropna()

	return data[['DeliveryDateTime', 'Price']]

# %%
def getDayAheadPrices():
	"""
	Intraday IDA2 Auction: Auction at 22:00 for every 15min interval in the next day

	Returns
	-------
	data : Pandas DataFrame with column datetime(CET) and electricity price[€/MWh]

	"""
	urls = ["https://www.bsp-southpool.com/day-ahead-trading-results-si.html?file=files/documents/trading/MarketResultsAuction_2022.xlsx&cid=1291",
			"https://www.bsp-southpool.com/day-ahead-trading-results-si.html?file=files/documents/trading/MarketResultsAuction_2023.xlsx&cid=1291",
			"https://www.bsp-southpool.com/day-ahead-trading-results-si.html?file=files/documents/trading/MarketResultsAuction.xlsx&cid=1291"]

	data_merge = pd.DataFrame()

	for url in urls:
		# Download the Excel file
		response = requests.get(url)
		response.raise_for_status()

		# Load the Excel file into a pandas DataFrame
		excel_file = BytesIO(response.content)
		sheets = pd.read_excel(excel_file, sheet_name=None, skiprows=1)  # Load all sheets
		limited_sheets = [sheet.head(32) for sheet in sheets.values()]

		# Concatenate the sheets into a single DataFrame
		data = pd.concat(limited_sheets, ignore_index=True)
		data.rename(columns={'Delivery Date': 'date'}, inplace=True)

		data.index = data['date']

		data = data.iloc[:-1, 3:-1]
		data.columns = [i for i in range(0, 24)]
		data = pd.melt(data, value_vars=[i for i in range(0, 24)],
					var_name='Hour_q', value_name='Price', ignore_index=False)
		
		data = data.dropna()

		data.index = pd.to_datetime(data.index) + pd.to_timedelta(data['Hour_q'], unit='Hour')
		data.sort_index()

		data.index = data.index.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')

		data_merge = pd.concat([data_merge, data])

	return data_merge

# %%
def getDayAheadVolumes():
	"""
	DayAhead Auction: Auction at 12:00 for hours in the next day

	Returns
	-------
	data : Pandas DataFrame with column DateTime(CET) and electricity price[€/MWh] 

	"""
	urls = ["https://www.bsp-southpool.com/day-ahead-trading-results-si.html?file=files/documents/trading/MarketResultsAuction_2022.xlsx&cid=1291",
			"https://www.bsp-southpool.com/day-ahead-trading-results-si.html?file=files/documents/trading/MarketResultsAuction_2023.xlsx&cid=1291",
			"https://www.bsp-southpool.com/day-ahead-trading-results-si.html?file=files/documents/trading/MarketResultsAuction.xlsx&cid=1291"]

	data_merge = pd.DataFrame()

	# Download the Excel file
	for url in urls:
		response = requests.get(url)
		response.raise_for_status()

		# Load the Excel file into a pandas DataFrame
		excel_file = BytesIO(response.content)
		sheets = pd.read_excel(excel_file, sheet_name=None, skiprows=1)  # Load all sheets
		limited_sheets = [sheet.tail(31) for sheet in sheets.values()]

		# Concatenate the sheets into a single DataFrame
		data = pd.concat(limited_sheets, ignore_index=True)
		data.rename(columns={'Delivery Date': 'date'}, inplace=True)

		data.index = data['date']

		data = data.iloc[:-1, 3:-1]
		data.columns = [i for i in range(0, 24)]
		data = pd.melt(data, value_vars=[i for i in range(0, 24)],
					var_name='Hour_q', value_name='Price', ignore_index=False)

		data = data.dropna()
		data.drop('Delivery Date', inplace = True)

		data.index = pd.to_datetime(data.index) + pd.to_timedelta(data['Hour_q'], unit='Hour')
		data.sort_index()

		data.index = data.index.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')

		data_merge = pd.concat([data_merge, data])

	return data_merge

def getIDA1Prices():
    """
    Intraday IDA1 Auction: Auction at 15:00 for every 15min interval in the next day

    Returns
    -------
    data : Pandas DataFrame with column datetime(CET) and electricity price[€/MWh]

    """
    url = "https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MarketResultsAuction_IDA1.xlsx&cid=1289"
    
    # Download the Excel file
    response = requests.get(url)
    response.raise_for_status()
    
    # Load the Excel file into a pandas DataFrame
    excel_file = BytesIO(response.content)
    sheets = pd.read_excel(excel_file, sheet_name=None, skiprows=1)  # Load all sheets
    limited_sheets = [sheet.head(34) for sheet in sheets.values()]

    # Concatenate the sheets into a single DataFrame
    data = pd.concat(limited_sheets, ignore_index=True)

    data = pd.melt(data, id_vars=['Delivery Date'], value_vars=[i for i in range(1, 97)],
                    var_name='Hour_q', value_name='Price')

    data['Hour_q']=data['Hour_q']-1
    # Convert the Delivery Date to datetime
    data['Delivery Date'] = pd.to_datetime(data['Delivery Date'])
    data['DeliveryDateTime'] = pd.to_datetime(data['Delivery Date']) + pd.to_timedelta(data['Hour_q'], unit='Minute')*15
    
    # Drop unnecessary columns
    data = data[['DeliveryDateTime', 'Price']]
    # Localize to CET (Central European Time)
    data['DeliveryDateTime'] = data['DeliveryDateTime'].dt.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')
    data=data.dropna()
    return data

# %%
def getIDA2Prices():
    """
    Intraday IDA2 Auction: Auction at 22:00 for every 15min interval in the next day

    Returns
    -------
    data : Pandas DataFrame with column datetime(CET) and electricity price[€/MWh]

    """
    url = "https://www.bsp-southpool.com/spajanje-trgov-ida.html?file=files/documents/trading/MarketResultsAuction_IDA2.xlsx&cid=3151"
    
    # Download the Excel file
    response = requests.get(url)
    response.raise_for_status()
    
    # Load the Excel file into a pandas DataFrame
    excel_file = BytesIO(response.content)
    sheets = pd.read_excel(excel_file, sheet_name=None, skiprows=1)  # Load all sheets
    limited_sheets = [sheet.head(34) for sheet in sheets.values()]

    # Concatenate the sheets into a single DataFrame
    data = pd.concat(limited_sheets, ignore_index=True)

    data = pd.melt(data, id_vars=['Delivery Date'], value_vars=[i for i in range(1, 97)],
                    var_name='Hour_q', value_name='Price')

    data['Hour_q']=data['Hour_q']-1
    # Convert the Delivery Date to datetime
    data['Delivery Date'] = pd.to_datetime(data['Delivery Date'])
    data['DeliveryDateTime'] = pd.to_datetime(data['Delivery Date']) + pd.to_timedelta(data['Hour_q'], unit='Minute')*15
    
    # Drop unnecessary columns
    data = data[['DeliveryDateTime', 'Price']]
    # Localize to CET (Central European Time)
    data['DeliveryDateTime'] = data['DeliveryDateTime'].dt.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')
    data=data.dropna()
    return data

def getIDA2_Old():
	"""
	Intraday IDA2 Auction: Auction at 22:00 for every 15min interval in the next day

	Returns
	-------
	data : Pandas DataFrame with column datetime(CET) and electricity price[€/MWh]

	"""
	urls = ["https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MI2_MarketResultsAuction.xlsx&cid=3190",
			"https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MI2_MarketResultsAuction_2023.xlsx&cid=3190",
			"https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MI2_MarketResultsAuction_2022.xlsx&cid=3190"]

	data_merge = pd.DataFrame()

	for url in urls:
		# Download the Excel file
		response = requests.get(url)
		response.raise_for_status()

		# Load the Excel file into a pandas DataFrame
		excel_file = BytesIO(response.content)
		sheets = pd.read_excel(excel_file, sheet_name=None, skiprows=1)  # Load all sheets
		limited_sheets = [sheet.head(32) for sheet in sheets.values()]

		# Concatenate the sheets into a single DataFrame
		data = pd.concat(limited_sheets, ignore_index=True)
		data.rename(columns={'Delivery Date': 'date'}, inplace=True)

		data.index = data['date']

		data = data.iloc[:-1, 2:26]
		data.columns = [i for i in range(0, 24)]
		data = pd.melt(data, value_vars=[i for i in range(0, 24)],
					var_name='Hour_q', value_name='Price', ignore_index=False)

		data = data.dropna()

		data.index = pd.to_datetime(data.index) + pd.to_timedelta(data['Hour_q'], unit='Hour')
		data.sort_index()

		data.index = data.index.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')

		data_merge = pd.concat([data_merge, data])

	return data_merge

def getIDA1_Old():
	"""
	Intraday IDA2 Auction: Auction at 22:00 for every 15min interval in the next day

	Returns
	-------
	data : Pandas DataFrame with column datetime(CET) and electricity price[€/MWh]

	"""
	urls = ["https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MI1_MarketResultsAuction.xlsx&cid=1289",
			"https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MI1_MarketResultsAuction_2023.xlsx&cid=1289",
			"https://www.bsp-southpool.com/market-data/intraday-auction.html?file=files/documents/trading/MI1_MarketResultsAuction_2022.xlsx&cid=1289"]

	data_merge = pd.DataFrame()

	for url in urls:
		# Download the Excel file
		response = requests.get(url)
		response.raise_for_status()

		# Load the Excel file into a pandas DataFrame
		excel_file = BytesIO(response.content)
		sheets = pd.read_excel(excel_file, sheet_name=None, skiprows=1)  # Load all sheets
		limited_sheets = [sheet.head(32) for sheet in sheets.values()]

		# Concatenate the sheets into a single DataFrame
		data = pd.concat(limited_sheets, ignore_index=True)
		data.rename(columns={'Delivery Date': 'date'}, inplace=True)

		data.index = data['date']

		data = data.iloc[:-1, 2:26]
		data.columns = [i for i in range(0, 24)]
		data = pd.melt(data, value_vars=[i for i in range(0, 24)],
					var_name='Hour_q', value_name='Price', ignore_index=False)

		data = data.dropna()

		data.index = pd.to_datetime(data.index) + pd.to_timedelta(data['Hour_q'], unit='Hour')
		data.sort_index()

		data.index = data.index.tz_localize('Europe/Ljubljana', ambiguous='NaT', nonexistent='NaT')

		data_merge = pd.concat([data_merge, data])

	return data_merge
