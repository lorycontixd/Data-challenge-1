#main.py
import matplotlib.pyplot as plt
import pandas as pd
import csv
from itertools import zip_longest
from functions import *

path_source = "./source/"
path_worked = "./worked/"

#a = zip(*csv.reader(open(path_worked+"c_us.csv", "r+")))
#csv.writer(open(path_worked+"confirmed_us.csv", "w+")).writerows(a)

#create empty dataframes
df_italy = pd.DataFrame(columns=['Date','Confirmed','Deaths','Recovers'])
df_us = pd.DataFrame(columns=['Date','Confirmed','Deaths','Recovers'])
df_spain = pd.DataFrame(columns=['Date','Confirmed','Deaths','Recovers'])

df_italy = LoadData(df_italy,'italy')
df_us = LoadData(df_us,'us')
df_spain = LoadData(df_spain,'spain')
dataframes = [df_italy,df_us,df_spain]

GraphDF(dataframes,'deaths')



#print(df_spain)
