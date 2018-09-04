import pandas as pd
import numpy as np
import matplotlib as mp
import datetime
from datetime import datetime
import more_itertools as mit
import statistics as stat
import math

StartCapital = 10000
Commissions = 0.0050

def DateCheck(x):
    try:
        x["Date"] = pd.to_datetime(x["Date"], format="%Y-%m-%d")
    except:
        x["Date"] = pd.to_datetime(x["Date"], format="%m/%d/%Y")

# start = "2010-02-11"
# end = "2018-06-04" #datetime.today().date()
# TQQQBase = web.DataReader("TQQQ", "morningstar", start, end)
# TQQQBase.to_csv("TQQQ"+".csv")

VIXBase = pd.read_csv("^VIX.csv")
VXVBase = pd.read_csv("^VXV.csv")
TQQQBase = pd.read_csv("TQQQ.csv")
DateCheck(VIXBase)
DateCheck(VXVBase)
DateCheck(TQQQBase)

TQQQBase = TQQQBase.drop("Adj Close",1)
TQQQBase = TQQQBase.drop("Volume",1)

ORatio = []
for i in range(0, len(VIXBase["Date"])):
    ORatio.append(VIXBase["Open"].iloc[i]/VXVBase["Open"].iloc[i])

ATR =[]
for i in range(0,len(TQQQBase["Date"])):
    if i>=10 and len(ATR)>=10:
        curTQQQ = TQQQBase.head(i)
        curTQQQ = curTQQQ.tail(10)
        ATR.append(np.average(curTQQQ["High"]-curTQQQ["Low"].tolist()))
    else:
        ATR.append(0)

Ratio = []
Stop = []
CAGR = []
StDev = []
DrawDown = []
Sharpe = []
MaR = []
SM = []

for MaxRatio in mit.numeric_range(0.88, 1.21, 0.001):
    for MaxStop in mit.numeric_range(0, 5, 0.01):
        Enter = []
        Capital = []
        Shares = []
        DayCng = []
        High= []

        MaxRatio = round(MaxRatio,3)
        MaxStop = round(MaxStop,2)

        for i in range(0, len(ORatio)):
            if MaxRatio < ORatio[i] or ATR[i] == 0:
                Enter.append(0)
            elif MaxRatio > ORatio[i] and TQQQBase["Open"][i]-MaxStop*ATR[i] >= TQQQBase["Low"][i]:
                Enter.append(-1)
            else:
                Enter.append(1)
        #Блок расчёта капитала
        for i in range (0, len(Enter)):
            #Если самая первая строчка
            if i == 0:
                Shares.append(0)
                Capital.append(StartCapital)
            #Если вчера ещё были в позиции, а сегодня надо выходить
            elif Enter[i] == 0 and Enter[i-1] == 1:
                Shares.append(0)
                Capital.append(Shares[i-1]*TQQQBase["Open"][i]-Shares[i-1]*Commissions)
            #Если вчера и сегодня нужно быть вне позиции
            elif Enter[i] == 0 and (Enter[i-1] == 0 or Enter[i-1] == -1):
                Shares.append(0)
                Capital.append(Capital[i-1])
            #Если получили стоп, но зашли в позу только сегодня или вчера получили стоп
            elif Enter[i] == -1 and (Enter[i-1] == 0 or Enter[i-1] == -1):
                Shares.append(Capital[i-1]/TQQQBase["Open"][i])
                Capital.append(Shares[i]*(TQQQBase["Open"][i]-MaxStop*ATR[i])-Shares[i]*Commissions*2)
            #Если получили стоп, но были в позе ранее
            elif Enter[i] == -1 and Enter[i-1] == 1:
                Shares.append(Shares[i-1])
                Capital.append(Shares[i]*(TQQQBase["Open"][i]-MaxStop*ATR[i])-Shares[i]*Commissions)
            #Если вчера были вне позици, а сегодня надо входить
            elif Enter[i] == 1 and (Enter[i-1] == 0 or Enter[i-1] == -1):
                Shares.append(Capital[i-1]/TQQQBase["Open"][i])
                Capital.append(Capital[i-1]-Shares[i]*Commissions)
            #Если сегодня спокойно сидим в позиции
            elif Enter[i] == 1:
                Shares.append(Shares[i-1])
                Capital.append(Shares[i]*TQQQBase["Open"][i])

        TQQQBase["Shares"] = Shares
        TQQQBase["Capital"] = Capital

        for i in range(0, len(TQQQBase["Capital"])):
            if i == 0:
                DayCng.append(0)
            else:
                DayCng.append(TQQQBase["Capital"].iloc[i]/TQQQBase["Capital"].iloc[i-1])
        TQQQBase["DayCng"] = DayCng

        for i in range(0, len(TQQQBase["Capital"])):
            if TQQQBase["Capital"]>High:
                High = TQQQBase["Capital"]


        Ratio.append(MaxRatio)
        Stop.append(MaxStop)
        CAGR.append(((TQQQBase["Capital"].iloc[-1]/TQQQBase["Capital"].iloc[0])**
                 (1/(TQQQBase["Date"].iloc[-1].year-0.5-TQQQBase["Date"].iloc[0].year))-1)*100)
        StDev.append(stat.stdev(TQQQBase["DayCng"])*math.sqrt(252))

        print(StDev)