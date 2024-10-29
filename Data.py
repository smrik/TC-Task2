import pandas as pd
import requests
from io import BytesIO

def getDayAheadPrices():
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