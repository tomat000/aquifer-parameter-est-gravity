# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:57:37 2025

@author: 13055
"""

""" 
program to process continuous head data on Escondida transect 

"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

df = pd.read_csv('ESC_E-01-A_20250422173224_DH.csv')
df['Date and time (UTC-07:00)'] =  pd.to_datetime(df['Date and time (UTC-07:00)'], format='%m/%d/%Y %I:%M:%S %p')
df.head()
fig, ax = plt.subplots()
n = 100
plt.plot(df['Date and time (UTC-07:00)'][:-3][::n],
         df['Water level referenced to Ground surface (ft)'][:-3][::n])


#%%
years = mdates.YearLocator()  # Every year
months = mdates.MonthLocator()  # Every month
years_fmt = mdates.DateFormatter('%Y-%m')

ax.xaxis.set_major_locator(months)  # Change major locator to months
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7)) #add day locator for minor ticks


plt.show()