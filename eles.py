# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
import time
from selenium.webdriver.common.keys import Keys
import pandas as pd

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
	df_current.to_csv("data/ELES.csv")

df_current = pd.read_csv("data/ELES.csv", index_col=0)


start_date = datetime(2022, 1, 1)
current_date = datetime.fromisoformat(df_current.index[-1])
end_date = datetime.today() + timedelta(days=1)
end_date = datetime.replace(end_date, hour=23, minute=0, second=0, microsecond=0)

if current_date < end_date:
	eles_update(current_date, end_date, df_current)
else:
	print("Data is up to date")