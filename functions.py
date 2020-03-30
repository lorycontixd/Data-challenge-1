#functions.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import optimize
from scipy import special
from scipy.interpolate import lagrange
from lmfit import Model
from lmfit.models import LinearModel, StepModel

path_source = "./source/"
path_worked = "./worked/"
myparameters = []

#FUNZIONI FIT
def FitLineare(a,b,x):
    return a*x+b
def FitEsponenziale(a,b,x):
    return a*math.exp(b*x)
def FitGaussiana(a,b,c,x):
    return a*math.exp(-(x-b)**2 / 2*c**2 )
def FitPol(g,x,*a):
    n = len(a)
    assert n==g+1
    assert g < 3
    if g==1:
        return a[0]*x + a[1]
    elif g==2:
        return a[0]* x**2 + a[1]*x + a[2]
def FitLagrange(x,y): #x,y arrays
    poly = lagrange(x, y)
    return poly
#FUNZIONE SIGMOIDEA
def FitSigmoidea(x,a,b,c):
    return a * math.exp(b+c*x) / b+math.exp(b+c*x)
def FitError(x,a,b,c):
    return a*math.erf(b*(x-c))
def FitHypTan(x,a,b,c):
    return a*math.tanh(b*(x-c))


def LoadData(df,country):
    countryname = country.lower()
    pathname = path_worked+countryname+'.csv'
    print("Loading data from: ",pathname)
    file = open(pathname,'r+')
    content = file.read()
    lines = content.split('\n')
    lines = lines[2:] #first line is country name
    for i in range(len(lines)-1):
        ll = lines[i].split(',')
        cols = {'Date':[ll[0]],'Confirmed':[ll[1]],'Deaths':[ll[2]],'Recovers':[ll[3]]}
        df_temp = pd.DataFrame(cols,columns=['Date','Confirmed','Deaths','Recovers'])
        df = df.append(df_temp,ignore_index = True)
    return df

def GraphDF(dfs,parameter):
    #checks
    for i in range(len(dfs)):
        assert len(dfs[0].index) == len(dfs[i].index)
    column = 0
    parameter = parameter.lower()
    accepted_parameters = ['confirmed','deaths','recovers']
    if parameter not in accepted_parameters:
        print("BadArgument: parameter not accepted")
        return
    else:
        if parameter == 'confirmed':
            column = 1
        elif parameter == 'deaths':
            column = 2
        elif parameter == 'recovers':
            column = 3
    n = [i for i in range(0,len(dfs[0].index))]
    matrix = [ ]
    for dataframe in dfs:
        temp = []
        for row in range(0,len(dataframe.index)):
            temp.append(dataframe.iloc[row,column])
        matrix.append(temp)

    #print(len(dataframe.index),len(matrix[0]))
    #print(len(dataframe.index),len(matrix[1]))
    #print(len(dataframe.index),len(matrix[2]))
    plt.plot(n,matrix[0])
    plt.plot(n,matrix[1])
    plt.plot(n,matrix[2])
    plt.show()

def LoadDataItalia():
    df = pd.DataFrame(columns = ['Data e Ora','Isolamento Domiciliare','Terapia Intensiva','Attualmente Positivi','Nuovi positivi','Guariti','Deceduti','Totale Casi','Tamponi'])
    sourcefile = './source/covid_italia.csv'
    file = open(sourcefile,'r+')
    content = file.read()
    lines = content.split('\n')
    lines = lines[1:]
    for i in range(len(lines)-1):
        ll = lines[i].split(',')
        cols = {'Data e Ora':[ll[0]] ,'Isolamento Domiciliare':[ll[5]], 'Terapia Intensiva':[ll[3]] ,'Attualmente Positivi':[ll[6]] , 'Nuovi positivi':[ll[7]] , 'Guariti':[ll[8]] , 'Deceduti':[ll[9]] , 'Totale Casi':[ll[10]] , 'Tamponi':[ll[11]]}
        df_temp = pd.DataFrame(cols,columns = ['Data e Ora','Isolamento Domiciliare','Terapia Intensiva','Attualmente Positivi','Nuovi positivi','Guariti','Deceduti','Totale Casi','Tamponi'])
        df = df.append(df_temp,ignore_index = True)
    return df

def Graph1(df,title,xlab,ylab,ylab2):
    n=len(df.index)
    days = [i for i in range(n)]
    dates = [str(df.iloc[i][0]) for i in range(0,n)]
    total_deaths = [int(df.iloc[i][6]) for i in range(0,n)] #list of deaths, bottom bar
    total_positives = [int(df.iloc[i][3]) for i in range(0,n)] #list of positives, middle bar
    total_recovered = [int(df.iloc[i][5]) for i in range(0,n)] #list of recovered, top bar
    total_home = [int(df.iloc[i][1]) for i in range(0,n)] #list of swabs (tamponi), scatter plot
    bars = np.add(total_positives,total_deaths).tolist() #heights of each bar
    #bars2 = np.asarray(bars)

    #print(total_positives)
    #print(n,len(total_positives))

    barwidth = 1
    edge_color = 'black'
    MyFitSigmoidea = np.vectorize(FitSigmoidea)
    init_par_sigmo = [2,-60,3]
    params, params_covariance = optimize.curve_fit(MyFitSigmoidea, days , bars , p0 = init_par_sigmo)
    #print(params)
    global myparameters
    myparameters = params

    f, ax = plt.subplots(figsize=(12,7))
    f.patch.set_facecolor('xkcd:peach')
    ax.set_facecolor('xkcd:white') #background color
    #bottom bar - deaths
    plt.scatter(days,total_home,color='xkcd:green',zorder=10,label='Isolamento Domiciliare')
    plt.plot(days,total_home,'g',linewidth=0.7)
    plt.plot(days,MyFitSigmoidea(days,params[0],params[1],params[2]),color='#F000FF' ,label = 'Sigmoid Fit')
    plt.bar(days,total_deaths,width=barwidth,color='#E20C0C',edgecolor=edge_color,label='Decessi')
    plt.text(3, 60000, 'p[0]: '+str(params[0]) , horizontalalignment='center' ,verticalalignment='center')
    plt.text(3, 63333, 'p[1]: '+str(params[1]) , horizontalalignment='center' ,verticalalignment='center')
    plt.text(3, 66666, 'p[2]: '+str(params[2]) , horizontalalignment='center' ,verticalalignment='center')
    #middle bar - positives
    p2 = plt.bar(days,total_positives,bottom=total_deaths,width=barwidth,color='#FFF000',edgecolor=edge_color,label='Attualmente positivi')
    #top bar
    p3 = plt.bar(days,total_recovered,bottom=bars,width=barwidth,color='#1383D5',edgecolor=edge_color,label='Guariti')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(ylab2)
    plt.title(title)
    plt.xticks(days, dates, fontweight='bold',rotation=90)

    ax.legend()
    plt.show()
