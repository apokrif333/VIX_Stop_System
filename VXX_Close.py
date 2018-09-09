import pandas as pd
import numpy as np
import matplotlib as mp
import datetime
from datetime import datetime
import more_itertools as mit
import statistics as stat
import math

StartCapital = 10000000
Commissions = 0.00017

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
VXXBase = pd.read_csv("VXX.csv")
DateCheck(VIXBase)
DateCheck(VXVBase)
DateCheck(VXXBase)

CRatio = []
for i in range(0, len(VIXBase["Date"])):
    CRatio.append(VIXBase["Close"].iloc[i]/VXVBase["Close"].iloc[i])

ATR =[]
for i in range(1, len(VXXBase["Date"])+1):
    if i >= 10:
        curVXX = VXXBase.head(i)
        curVXX = curVXX.tail(10)
        ATR.append(np.average(curVXX["High"]-curVXX["Low"].tolist()))
    else:
        ATR.append(0)

#Глобальные массивы, которые сохранятся после всех расчётов
Ratio = []
StopL = []
CAGR = []
StDev = []
DrawDown = []
Sharpe = []
MaR = []
SM = []

for MaxRatio in mit.numeric_range(0.88, 1.21, 0.001):
    for MaxStop in mit.numeric_range(0.0, 5, 0.01):
        #Массивы текущего блока расчётов
        Enter = []
        Stop = []
        EnterPrice = []
        Capital = []
        Shares = []
        Comm = []
        DayCng = []
        Down = []

        MaxRatio = round(MaxRatio,3)
        MaxStop = round(MaxStop,2)

        for i in range(0, len(CRatio)):
            if MaxRatio < CRatio[i] or ATR[i] == 0:
                Enter.append(0)
            else:
                Enter.append(1)
            if i > 0 and ATR[i-1] > 0:
                Stop.append(VXXBase["Close"][i-1]+MaxStop*ATR[i-1])
            else:
                Stop.append(0)

        #Блок расчёта капитала
        for i in range (0, len(Enter)):
            #Если самая первая строчка
            if i == 0:
                Capital.append(StartCapital)
                Comm.append(0)
                Shares.append(0)
                EnterPrice.append(0)
            #Если вчера ещё были в позиции, а сегодня надо выходить
            elif Enter[i] == 0 and Enter[i-1] == 1 and Stop[i] > VXXBase["High"][i]:
                Capital.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-VXXBase["Close"][i])*Shares[-1]) * (1-Commissions))
                Comm.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-VXXBase["Close"][i])*Shares[-1]) * Commissions)
                Shares.append(0)
                EnterPrice.append(0)
            #Если вчера и сегодня нужно быть вне позиции
            elif Enter[i] == 0 and Enter[i-1] == 0:
                Capital.append(Capital[-1])
                Comm.append(0)
                Shares.append(0)
                EnterPrice.append(0)
            # Если сегодня получили стоп и опять надо входить
            elif Enter[i] == 1 and Enter[i-1] == 1 and Stop[i] <= VXXBase["High"][i]:
                if Stop[i] <= VXXBase["Open"][i]:
                    Capital.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-VXXBase["Open"][i])*Shares[-1])
                                   * (1-2*Commissions))
                    Comm.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-VXXBase["Open"][i])*Shares[-1])
                                   *2*Commissions)
                else:
                    Capital.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-Stop[i])*Shares[-1])
                                   * (1-2*Commissions))
                    Comm.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-Stop[i])*Shares[-1])
                                   *2*Commissions)
                Shares.append(Capital[-1]/VXXBase["Close"][i])
                EnterPrice.append(VXXBase["Close"][i])
            #Если получили стоп, но были в позе ранее
            elif Stop[i] <= VXXBase["High"][i] and Enter[i-1] == 1:
                if Stop[i] <= VXXBase["Open"][i]:
                    Capital.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-VXXBase["Open"][i])*Shares[-1]) * (1-Commissions))
                    Comm.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-VXXBase["Open"][i])*Shares[-1]) * Commissions)
                else:
                    Capital.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-Stop[i])*Shares[-1])*(1-Commissions))
                    Comm.append((Shares[-1]*EnterPrice[-1] + (EnterPrice[-1]-Stop[i])*Shares[-1]) * Commissions)
                Shares.append(0)
                EnterPrice.append(0)
            #Если вчера были вне позици, а сегодня надо входить
            elif Enter[i] == 1 and Enter[i-1] == 0:
                Capital.append(Capital[-1]*(1-Commissions))
                Comm.append(Capital[-1]*Commissions)
                Shares.append(Capital[-1]/VXXBase["Close"][i])
                EnterPrice.append(VXXBase["Close"][i])
            #Если сегодня спокойно сидим в позиции
            elif Enter[i] == 1:
                Capital.append(Shares[-1]*EnterPrice[-1]+(EnterPrice[-1]-VXXBase["Close"][i])*Shares[-1])
                Comm.append(0)
                Shares.append(Shares[-1])
                EnterPrice.append(EnterPrice[-1])

        for i in range(0, len(Capital)):
            if i == 0:
                DayCng.append(0)
            else:
                DayCng.append(Capital[i]/Capital[i-1]-1)

        High = 0
        for i in range(0, len(Capital)):
            if Capital[i]>High:
                High = Capital[i]
            Down.append((Capital[i]/High-1)*100)

        Ratio.append(MaxRatio)
        StopL.append(MaxStop)
        CAGR.append(((Capital[-1]/Capital[0])**
                 (1/(VXXBase["Date"].iloc[-1].year-VXXBase["Date"].iloc[0].year))-1)*100)
        StDev.append(stat.stdev(DayCng)*math.sqrt(252))
        DrawDown.append(min(Down))
        Sharpe.append(CAGR[-1]/StDev[-1])
        MaR.append(abs(CAGR[-1]/DrawDown[-1]))
        SM.append(Sharpe[-1]*MaR[-1])

        print(MaxRatio)
        print(MaxStop)

exportTable = pd.DataFrame({"Ratio": Ratio,
                            "Stop": StopL,
                            "CAGR": CAGR,
                            "StDev": StDev,
                            "DrawDown": DrawDown,
                            "Sharpe": Sharpe,
                            "MaR": MaR,
                            "SM": SM},
                           columns=["Ratio", "Stop", "CAGR", "StDev", "DrawDown", "Sharpe", "MaR", "SM"]
                           )

exportTable.to_csv("C:/Users/Tom/Google Диск/Прог/VXX_Close_Metric.csv")