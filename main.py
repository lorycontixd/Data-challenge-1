import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from functions import *
cities = ['milano','bergamo','bologna','roma']

newcases = LoadNewCases('milano')
mydf = LoadDataItalia()

Graph1(mydf)
print("Saved graph1")
Graph2(mydf)
print("Saved graph2")
#Over65()
print("Saved graph over 65")
#Pollution()
print("Saved graph for pollution")

#GraphRegione()
print("Saved graph for regions")
Eta()
print("Saved graph for age")
print(" ")
Size()
print(" ")
print("Saved graph for city population vs cases")
