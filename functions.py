#functions.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import math
import matplotlib.ticker as ticker
from scipy import optimize
from scipy import special
from scipy.interpolate import lagrange
from lmfit import Model
from lmfit.models import LinearModel, StepModel


path_source = "./source/"
path_worked = "./worked/"
myparameters = []

cases = []

#FUNZIONI FIT
def FitLineare(a,b,x):
    return a*x+b
def FitEsponenziale(a,b,x):
    return a*math.exp(b*x)
def FitGaussiana(x,a,b,c):
    return a*math.exp( -(x-b)**2 / (2* c**2) )
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
    return a * math.exp(b+c*x) / 1+math.exp(b+c*x)
def FitError(x,a,b,c):
    return a*math.erf(b*(x-c))
def FitHypTan(x,a,b,c):
    return a*math.tanh(b*(x-c))
def FitEMG(x,a,b,c):
    y = (a/2)*math.exp( (a/2)*(2*b + a*c**2-2*x) ) * math.erfc( (b+a*c*+2-x)/(math.sqrt(2)*c) )
    return y

def LoadDataItalia():
    df = pd.DataFrame(columns = ['Date','Home isolation','Intensive Care','Currently Positive','New Positives','Recovers','Deaths','Total Cases','Swabs'])
    sourcefile = './source/covid_italia.csv'
    file = open(sourcefile,'r+')
    content = file.read()
    lines = content.split('\n')
    lines = lines[1:]
    for i in range(len(lines)-1):
        ll = lines[i].split(',')
        cols = {'Date':[ll[0]] ,'Home isolation':[ll[5]], 'Intensive Care':[ll[3]] ,'Currently Positive':[ll[6]] , 'New Positives':[ll[7]] , 'Recovers':[ll[8]] , 'Deaths':[ll[9]] , 'Total Cases':[ll[10]] , 'Swabs':[ll[11]]}
        df_temp = pd.DataFrame(cols,columns = ['Date','Home isolation','Intensive Care','Currently Positive','New Positives','Recovers','Deaths','Total Cases','Swabs'])
        df = df.append(df_temp,ignore_index = True)
    return df

def GraphRegione():
    dfs = []
    days = [5,12,19,26]
    regions = ['campania','piemonte','veneto','emilia','liguria','friuli']
    d = [424,422,267,199,286,153]
    i=1 #iterator
    for region in regions:
        df = pd.DataFrame(columns = ['Date','Region','Isolated','New Positives'])
        sourcefile = './worked/regioni/covid_'+region+'.csv'
        file = open(sourcefile,'r+')
        content = file.read()
        lines = content.split('\n')
        for i in range(len(lines)-1):
            if i!= 4 and i!=11 and i!=18 and i!= 25:
                continue
            ll = lines[i].split(',')
            cols = {'Date':[ll[0]],'Region':[ll[3]],'Isolated':[ll[9]],'New Positives':[ll[12]]}
            df_temp = pd.DataFrame(cols,columns = ['Date','Region','Isolated','New Positives'])
            df = df.append(df_temp,ignore_index = True)
        dfs.append(df)

    list0 = [int(dfs[0].iloc[i][3]) for i in range(len(dfs[0].index))]
    list1 = [int(dfs[1].iloc[i][3]) for i in range(len(dfs[1].index))]
    list2 = [int(dfs[2].iloc[i][3]) for i in range(len(dfs[2].index))]
    list3 = [int(dfs[3].iloc[i][3]) for i in range(len(dfs[3].index))]
    list4 = [int(dfs[4].iloc[i][3]) for i in range(len(dfs[4].index))]
    list5 = [int(dfs[5].iloc[i][3]) for i in range(len(dfs[5].index))]

    fig,ax = plt.subplots()
    plt.plot(days,list0,marker='.',label="Campania (424)")
    plt.plot(days,list4,marker='.',label="Liguria (286)")
    plt.plot(days,list2,marker='.',label="Veneto (267)")
    plt.plot(days,list3,marker='.',label="Emilia-Romagna (199)")
    plt.plot(days,list1,marker='.',label="Piemonte (172)")
    plt.plot(days,list5,marker='.',label="Friuli (153)")

    plt.title("Andamento nel tempo dei nuovi casi COVID in base alla densità di popolazione")
    plt.xlabel("Day of March")
    plt.ylabel("Number of cases")
    ax.legend()
    plt.show()






def LoadNewCases(city):
    city = city.lower()
    cities = ['milano','bergamo','roma','bologna']
    assert city in cities
    newcases = []
    sourcefile = './worked/useful/'+city+'.csv'
    file = open(sourcefile,'r+')
    content = file.read()
    lines = content.split('\n')
    for i in range(len(lines)-1):
        ll = lines[i].split(',')
        newcases.append(int(ll[6]))
    return newcases



def Graph1(df): #used for total cases
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
    init_par_sigmo = [10,-83,4]
    #init_par_tanh = [10,0.3,20]
    params, params_covariance = optimize.curve_fit(MyFitSigmoidea, days , bars , p0 = init_par_sigmo)
    #print(params)
    global myparameters
    myparameters = params

    f, ax = plt.subplots(figsize=(12,7))
    f.patch.set_facecolor('xkcd:peach')
    graph_title = 'General Data Analysis of newly spread virus COVID-19 in Italy'
    xlab = 'Date'
    ylab = 'Total number of cases'
    ylab2 = 'Home Isolation'
    ax.set_facecolor('xkcd:white') #background color

    #bottom bar - deaths
    plt.scatter(days,total_home,color='xkcd:green',zorder=10,label='Home Isolation')
    plt.plot(days,total_home,'g',linewidth=0.7)
    plt.plot(days,MyFitSigmoidea(days,params[0],params[1],params[2]),color='#F000FF' ,label = 'Sigmoid Fit')
    plt.bar(days,total_deaths,width=barwidth,color='#E20C0C',edgecolor=edge_color,label='Deaths')
    plt.text(3, 60000, 'p[0]: '+str(params[0]) , horizontalalignment='center' ,verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
    plt.text(3, 65000, 'p[1]: '+str(params[1]) , horizontalalignment='center' ,verticalalignment='center',bbox=dict(facecolor='orange', alpha=0.5))
    plt.text(3, 70000, 'p[2]: '+str(params[2]) , horizontalalignment='center' ,verticalalignment='center',bbox=dict(facecolor='yellow', alpha=0.5))
    #middle bar - positives
    p2 = plt.bar(days,total_positives,bottom=total_deaths,width=barwidth,color='#FFF000',edgecolor=edge_color,label='Currently Positive')
    #top bar
    p3 = plt.bar(days,total_recovered,bottom=bars,width=barwidth,color='#1383D5',edgecolor=edge_color,label='Recovers')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    secax = ax.secondary_yaxis('right')
    secax.set_ylabel(ylab2)
    plt.title(graph_title)
    plt.xticks(days, dates, fontweight='bold',rotation=90)

    ax.legend()
    plt.savefig("./images/graph1.png",bbox_inches='tight')

def Graph2(df): #used for new cases
    n=len(df.index)
    days = [i for i in range(n)]
    dates = [str(df.iloc[i][0]) for i in range(0,n)]
    new_positives = [int(df.iloc[i][4]) for i in range(0,n)]
    assert len(days) == len(new_positives)

    init = (1,4,1)
    MyG = np.vectorize(FitGaussiana)
    params,params_covariance = optimize.curve_fit(MyG,days,new_positives,p0=init)
    fig, ax = plt.subplots(figsize=(12,7))
    fig.patch.set_facecolor('xkcd:lavender')
    graph2_title = "Scaling of new positive cases in time"
    x2lab = 'Date'
    y2lab = 'Total number of cases'

    plt.plot(days,new_positives,label = 'New Positives',linewidth=2,marker='.')
    plt.plot(days,MyG(days,params[0],params[1],params[2]),label = 'Guassian Fit',linewidth=2)

    plt.text(0,4000, 'Eq: a*exp(-(x-b)**2 /(2* c**2))')
    plt.text(3, 3000, 'a: '+str(params[0]) , horizontalalignment='center' ,verticalalignment='center',bbox=dict(facecolor='red', alpha=0.5))
    plt.text(3, 3270, 'b: '+str(params[1]) , horizontalalignment='center' ,verticalalignment='center',bbox=dict(facecolor='orange', alpha=0.5))
    plt.text(3, 3540, 'c: '+str(params[2]) , horizontalalignment='center' ,verticalalignment='center',bbox=dict(facecolor='yellow', alpha=0.5))
    plt.xticks(days, dates, fontweight='bold',rotation=90)
    plt.title(graph2_title)
    plt.xlabel(x2lab)
    plt.ylabel(y2lab)
    ax.legend(loc='upper left')
    plt.savefig("./images/graph2.png",bbox_inches='tight')

#"METRO_ID","Metropolitan areas","VAR","Variables","TIME","Year","Unit Code","Unit","PowerCode Code","PowerCode","Reference Period Code","Reference Period","Value","Flag Codes","Flags"
def Over65():
    all_elders = []
    accepted_cities = ['bergamo','bologna','roma']
    for city in accepted_cities:
        #load DataFrame
        df = pd.DataFrame(columns = ['Area','Year','Variable','Value'])
        sourcefile = './worked/data/national/over65_'+city+'.csv'
        file = open(sourcefile,'r+')
        content = file.read()
        lines = content.split('\n')
        for i in range(len(lines)-1):
            ll = lines[i].split(',')
            cols = {'Area':[ll[1]],'Year':[ll[4]],'Variable':[ll[3]],'Value':[ll[12]]}
            df_temp = pd.DataFrame(cols,columns = ['Area','Year','Variable','Value'])
            df = df.append(df_temp,ignore_index = True)
        elders = [float(df.iloc[i][3]) for i in range(0,len(df.index)) if 'Elderly' in str(df.iloc[i][2])]
        if city != 'bergamo':
            elders.pop(0)
        days = [i for i in range(0,len(elders))]
        assert len(days) == len(elders)
        years = [int(df.iloc[i][1]) for i in range(0,len(df.index)) if 'Elderly' in str(df.iloc[i][2])]
        empty = [" " for i in range(0,len(df.index)) if 'Elderly' in str(df.iloc[i][2])]
        all_elders.append(elders)
    #print(all_elders,'\n')
    fig,ax = plt.subplots(3,figsize=(12,7))
    fig.patch.set_facecolor('xkcd:beige')
    fig.suptitle("Percentage of 65+ people on a city's population'")
    i=0
    colors = ['b','g','r']
    for list in all_elders:
        assert len(days) ==len(list)
        plt.sca(ax[i])
        plt.title(accepted_cities[i])
        plt.plot(days,list,color=colors[i],label=str(accepted_cities[i]))
        plt.legend(loc="upper right")
        if i == 2:
            plt.xticks(days, years , fontweight='bold',rotation=90)
            plt.xlabel('Year')
        else:
            plt.xticks(days, empty , fontweight='bold',rotation=90)
        i+=1
    for ax in ax.flat:
        ax.set(ylabel='Percentage (%)')
    plt.savefig("./images/over65.png",bbox_inches='tight')
    #plt.show(plt.xticks(days, years, fontweight='bold',rotation=90))

def Pollution():
    all_values = []
    cities = ['milano','bergamo','bologna','roma']
    for city in cities:
        df = pd.DataFrame(columns = ['Area','Year','Variable','Value'])
        sourcefile = './worked/data/national/pollution_'+city+'.csv'
        file = open(sourcefile,'r+')
        content = file.read()
        lines = content.split('\n')
        for i in range(len(lines)-1):
            ll = lines[i].split(',')
            cols = {'Area':[ll[1]],'Year':[ll[4]],'Variable':'Pollution in µg/m³','Value':[ll[12]]}
            df_temp = pd.DataFrame(cols,columns = ['Area','Year','Variable','Value'])
            df = df.append(df_temp,ignore_index = True)
        days = [i for i in range(len(df.index))]
        values = [float(df.iloc[i][3]) for i in range(len(df.index))]
        years = [df.iloc[i][1] for i in range(len(df.index))]
        empty = ["" for i in range(len(df.index))]
        colors = ['b','g','r']
        all_values.append(values)

    i=0
    fig,ax = plt.subplots(4,figsize=(12,8))
    fig.patch.set_facecolor('xkcd:peach')
    fig.suptitle("Average emission per year (µg/m³)")
    colors = ['b','g','r','y']
    for list in all_values:
        plt.sca(ax[i])
        plt.title(cities[i])
        plt.plot(days,list,color=colors[i],label=str(cities[i]))
        plt.legend(loc="upper right")
        if i == 3:
            plt.xticks(days, years , fontweight='bold',rotation=90)
            plt.xlabel('Year')
        else:
            plt.xticks(days, empty , fontweight='bold',rotation=90)
        i+=1
    fig.text(0.06, 0.5, "Average emission (µg/m³)", ha='center', va='center', rotation='vertical')
    plt.savefig("./images/emissions.png",bbox_inches='tight')

def Temperature(cases):
    cities = ['milano','bergamo','bologna','genova']
    city = cities[0]
    sourcefile = './worked/data/national/meteo/'+city+'-marzo.csv'
    df = pd.DataFrame(columns = ['Date','Temp.Ave (°C)','Temp.Min (°C)','Temp.Max (°C)','Humidity (%)','Wind.Ave (km/s)'])
    file = open(sourcefile,'r+')
    content = file.read()
    lines = content.split('\n')
    for i in range(len(lines)-1):
        ll = lines[i].split(';')
        cols = {'Date':[ll[1]],'Temp.Ave (°C)':[ll[2]],'Temp.Min (°C)':[ll[3]],'Temp.Max (°C)':[ll[4]],'Humidity (%)':[ll[6]],'Wind.Ave (km/s)':[ll[8]]}
        df_temp = pd.DataFrame(cols,columns = ['Date','Temp.Ave (°C)','Temp.Min (°C)','Temp.Max (°C)','Humidity (%)','Wind.Ave (km/s)'])
        df = df.append(df_temp,ignore_index = True)
    df.drop(df.tail(2).index,inplace=True)
    se = pd.Series(cases)
    df['Cases'] = se.values
    temp_ave = [int(df.iloc[i][1]) for i in range(len(df.index))]
    date = [str(df.iloc[i][0]) for i in range(len(df.index))]

    plt.scatter(temp_ave,cases)
    plt.show()

def Eta():
    età = [5,15,25,35,45,55,65,75,85,95]
    perc_tot = [0,0,0,0.1,0.1,0.6,2.7,9.6,16.6,19.0]
    perc_uomini = [0,0,0,0.2,0,0.7,3.3,11.3,19.3,25.8]
    perc_donne = [0,0,0,0,0.1,0.4,1.6,6.4,13.0,15.2]
    #print(len(perc_uomini)," , ",len(perc_donne)," , ",len(perc_tot))
    assert len(perc_tot)==len(perc_uomini)
    assert len(perc_tot)==len(perc_donne)
    x = [0,10,20,30,40,50,60,70,80,90]
    #x2 = [5,15,25,35,45,55,65,75,85,95]
    labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-80','90+']
    df = pd.DataFrame({'men':perc_uomini,'women':perc_donne,'tot':perc_tot},index=labels)

    fig,ax = plt.subplots(figsize=(11,7))
    for i, v in enumerate(perc_tot):
        plt.text(x[i] + 0.1, v + 0.01, str(v))

    ax.set_xlabel('Age', fontdict=dict(weight='bold'))
    ax.set_ylabel('Death Percentage for given age range (%)', fontdict=dict(weight='bold'))
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation='horizontal',ha='left')
    plt.bar(età,perc_uomini,width=2,label='Men',align='edge',color='blue')
    plt.bar(età,perc_donne,width=-2,label='Women',align='edge',color='green')
    plt.bar(età,perc_tot,width=2,label='Total',align='center',color='red')
    ax.legend()
    plt.title("Mortality rate for age ranges (In Italy)")
    plt.savefig('./images/age_mortality.png',bbox_inches='tight')

def Size():
    df = pd.DataFrame(columns = ['Country','Region','City','Population','Cases'])
    sourcefile = './worked/casi_popolazione.csv'
    file = open(sourcefile,'r+')
    content = file.read()
    lines = content.split('\n')
    for i in range(len(lines)-1):
        ll = lines[i].split(',')
        cols = {'Country':[ll[0]],'Region':[ll[1]],'City':[ll[2]],'Population':[ll[3]],'Cases':[ll[4]]}
        df_temp = pd.DataFrame(cols,columns = ['Country','Region','City','Population','Cases'])
        df = df.append(df_temp,ignore_index = True)
    sud_regions = ['Campania','Puglia','Lazio','Sicilia','Basilicata','Abruzzo','Calabria','Molise','Umbria']
    pop_nord = [int(df.iloc[i][3]) for i in range(len(df.index)) if df.iloc[i][0] == 'Italy' if str(df.iloc[i][1]) not in sud_regions]
    pop_sud = [int(df.iloc[i][3]) for i in range(len(df.index)) if df.iloc[i][0] == 'Italy' if str(df.iloc[i][1]) in sud_regions]
    pop_de = [int(df.iloc[i][3]) for i in range(len(df.index)) if df.iloc[i][0] == 'Germany']
    maxs = [int(max(pop_nord)),int(max(pop_de)),int(max(pop_sud))]
    mins = [int(min(pop_nord)),int(min(pop_de)),int(max(pop_sud))]
    pmax = max(maxs)
    pmin = min(mins)

    delta = (pmax-pmin)/ len(df.index)
    xpos = [i*delta for i in range(len(df.index))]
    cases_nord = [int(df.iloc[i][4]) for i in range(len(df.index)) if df.iloc[i][0] == 'Italy' if df.iloc[i][1] not in sud_regions]
    cases_sud = [int(df.iloc[i][4]) for i in range(len(df.index)) if df.iloc[i][0] == 'Italy' if df.iloc[i][1] in sud_regions]
    cases_de = [int(df.iloc[i][4]) for i in range(len(df.index)) if df.iloc[i][0] == 'Germany']
    assert len(pop_nord)+len(pop_de)+len(pop_sud)==len(cases_nord)+len(cases_de)+len(cases_sud)

    MyFitLineare = np.vectorize(FitLineare)
    fig,ax = plt.subplots(figsize=(12,8))
    params, params_covariance = optimize.curve_fit(MyFitLineare, pop_nord , cases_nord , p0 = [300,2])
    params2, params2_covariance = optimize.curve_fit(MyFitLineare, pop_sud , cases_sud , p0 = [140,3])
    params3, params3_covariance = optimize.curve_fit(MyFitLineare, pop_de , cases_de , p0 = [100,1])

    plt.title("Confirmed cases vs city size (Updated to 1/4/20)")
    plt.xlabel('Population (mln)',fontdict=dict(weight='bold'))
    plt.xticks(rotation=90)
    ax.set_xticks(xpos)
    plt.ylabel('Confirmed cases',fontdict=dict(weight='bold'))

    plt.scatter(pop_nord,cases_nord,color='red',label="Italian Northern City")
    plt.scatter(pop_de,cases_de,color='blue',label="German City")
    plt.scatter(pop_sud,cases_sud,color='orange',label="Italian Southern City")

    plt.plot(pop_nord,MyFitLineare(pop_nord,params[0],params[1]) ,color='red',label='Italian Northern Plot')
    plt.plot(pop_sud,MyFitLineare(pop_sud,params2[0],params2[1]) ,color='orange',label='Italian Souther Plot')
    plt.plot(pop_de,MyFitLineare(pop_de,params3[0],params3[1]) ,color='blue',label='German Plot')

    print("N-it: ",params[0],params[1])
    print("S-it: ",params2[0],params2[1])
    print("De: ",params3[0],params3[1])
    ax.text(10,7500,"-> Gradient N.Italy: "+str(params[0]),style='normal',color='red')
    ax.text(10,7200,"-> Gradient S.Italy: "+str(params2[0]),style='normal',color='orange')
    ax.text(10,6900,"-> Gradient Germany: "+str(params3[0]),style='normal',color='blue')
    ax.annotate('Milan',(pop_nord[0],cases_nord[0]))
    ax.annotate('Bergamo',(pop_nord[1],cases_nord[1]))
    ax.annotate('Cremona',(pop_nord[17],cases_nord[17]))
    ax.annotate('Munich',(pop_de[0],cases_de[0]))
    ax.annotate('Rome',(pop_sud[1],cases_sud[1]))
    ax.legend()
    plt.savefig('./images/population_cases.png',bbox_inches='tight')
