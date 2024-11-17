import Data as Data
import pandas as pd
from io import BytesIO
import collections
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import time
from selenium.webdriver.common.keys import Keys

# southpool data
# Day Ahead Auction
# day ahead trading results 
dap = Data.getDayAheadPrices()
dap.to_csv('data/da/day_ahead_prices.csv', index=True)
dav = Data.getDayAheadVolumes()
dav.to_csv('data/da/day_ahead_volumes.csv', index=True)

# Day Ahead Auction
# market coupling results
# --- missing ---

# Intraday Auction

#
# IDA 1
#
# ida1_old = Data.getIDA1_Old()
# ida1_old['DeliveryDateTime'] = ida1_old.index
# ida1_old.to_csv('data/ida1_old_prices.csv', index=False)
ida1_old = pd.read_csv('data/ida/ida1_old_prices.csv')
ida1 = Data.getIDA1Prices()
ida1 = pd.concat([ida1_old, ida1])
ida1.to_csv('data/ida/ida1_prices.csv', index=True)

# 
# IDA 2
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

# %%
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