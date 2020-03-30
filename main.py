import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from functions import *

mydf = LoadDataItalia()
#print(mydf)
graph_title = 'Analisi dati COVID-19 Italia'
xlab = 'Data'
ylab = 'Numero totale'
ylab2 = 'Isolamento Domiciliare'

Graph1(mydf,graph_title,xlab,ylab,ylab2)
print(myparameters)
