import pandas as pd
import numpy as np
import more_itertools as mit
import statistics as stat
import math
import time
import cmath
import os
import csv

from alpha_vantage.timeseries import TimeSeries
from yahoofinancials import YahooFinancials
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd

# Const
COMM = 0.0055 # Коэффициент размещения денег в одну сторону на 40$ цены тикера
CONST_COMM = 0.55 # Статичная комиссия, если объём меньше 100 акций
SLIPP = 0.015 # При каждом стопе проскальзывание 1,5 цента на акцию
ALPHA_KEY = 'FE8STYV4I7XHRIAI'

# Variables
analysis_tickers = ['ZIV'] # Чтобы скачать с yahoo, нужно выставить время в компьютере NY
start_cap = 10000
style = 'open'.lower() # open - выстраивает логику с открытия сессии, close, выстраивает логику на закрытие

download_data = True # Качает сплитованные данные с yahoo и не сплитованные с alpha. На alpha задержка 15 сек
start_date = datetime(1990, 1, 1) # Для yahoo, alpha выкачает всю доступную историю
end_date = datetime.now()
default_data_dir = 'exportTables' # Директория

# Globals
Year = []
BestRatio = []
BestStop = []

alpha_count = 0

# Формат даты в сторку
def dt_str(date: datetime) -> str:
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)


# Формат строки в дату
def str_dt(date: str) -> datetime:
    return datetime.strptime(date, '%Y-%m-%d')


# Формат листа со строками в лист с датами
def st_date(x):
    x["Date"] = pd.to_datetime(x["Date"])


# Формат цен
def form_price(n) -> float:
    return round(float(n), 2)


# Формат объёма
def form_volume(n) -> int:
    return int(round(float(n), 0))


# Не пустой ли объект
def empty_obj(n) -> bool:
    return n is not None and n != 0 and not cmath.isnan(n)


# Словарь с ценами
def dic_with_prices(prices: dict, ticker: str, date: datetime, open, high, low, close, volume):
    if date.weekday() > 5:
        print(f'Найден выходной в {ticker} на {date}')
        return

    open = form_price(open)
    high = form_price(high)
    low = form_price(low)
    close = form_price(close)
    volume = form_volume(volume)

    error_price = (not empty_obj(open)) or (not empty_obj(high)) or (not empty_obj(low)) or (not empty_obj(close))
    error_vol = not empty_obj(volume)

    if error_price:
        print(f'В {ticker} на {date} имеются пустые данные')
        return
    if error_vol:
        print(f'В {ticker} на {date} нет объёма')

    prices[date] = [open, high, low, close, volume]


# Сохраняем csv файл
def save_csv(base_dir: str, ticker: str, data: pd.DataFrame, source: str):
    path = os.path.join(base_dir)
    if not os.path.exists(path):
        os.makedirs(path)

    if source == 'alpha':
        print(f'{ticker} работает с альфой')
        path = os.path.join(path, ticker + ' NonSplit' + '.csv')
    elif source == 'yahoo':
        print(f'{ticker} работает с яху')
        path = os.path.join(path, ticker + '.csv')
    elif source == 'new_file':
        print(f'Сохраняем файл с тикером {ticker}')
        path = os.path.join(path, ticker + '.csv')
    else:
        print(f'Неопознанный источник данных для {ticker}')

    data.to_csv(path, index_label='Date')


# Загружаем csv файл
def load_csv(ticker: str, base_dir: str=default_data_dir) -> pd.DataFrame:
    path = os.path.join(base_dir, str(ticker) + '.csv')
    file = pd.read_csv(path)
    st_date(file)
    return file


# Скачиваем нужные тикеры из альфы
def download_alpha(ticker: str, base_dir: str = default_data_dir) -> pd.DataFrame:
    data = None
    global alpha_count

    try:
        ts = TimeSeries(key=ALPHA_KEY, retries=0)
        data, meta_data = ts.get_daily(ticker, outputsize='full')
    except Exception as err:
        if 'Invalid API call' in str(err):
            print(f'AlphaVantage: ticker data not available for {ticker}')
            return pd.DataFrame({})
        elif 'TimeoutError' in str(err):
            print(f'AlphaVantage: timeout while getting {ticker}')
        else:
            print(f'AlphaVantage: {err}')

    if data is None or len(data.values()) == 0:
        print('AlphaVantage: no data for %s' % ticker)
        return pd.DataFrame({})

    prices = {}
    for key in sorted(data.keys(), key=lambda d: datetime.strptime(d, '%Y-%m-%d')):
        secondary_dic = data[key]
        date = datetime.strptime(key, '%Y-%m-%d')
        dic_with_prices(prices, ticker, date, secondary_dic['1. open'], secondary_dic['2. high'],
                        secondary_dic['3. low'], secondary_dic['4. close'], secondary_dic['5. volume'])

    frame = pd.DataFrame.from_dict(prices, orient='index', columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    save_csv(base_dir, ticker, frame, 'alpha')
    time.sleep(15 if alpha_count != 0 else 0)
    alpha_count += 1


# Скачиваем тикеры из яху
def download_yahoo(ticker: str, base_dir: str = default_data_dir) -> pd.DataFrame:
    try:
        yf = YahooFinancials(ticker)
        data = yf.get_historical_stock_data(dt_str(start_date), dt_str(end_date), 'daily')
    except Exception as err:
        print(f'Unable to read data for {ticker}: {err}')
        return pd.DataFrame({})

    if data.get(ticker) is None or data[ticker].get('prices') is None or \
            data[ticker].get('timeZone') is None or len(data[ticker]['prices']) == 0:
        print(f'Yahoo: no data for {ticker}')
        return pd.DataFrame({})

    prices = {}
    for rec in sorted(data[ticker]['prices'], key=lambda r: r['date']):
        if rec.get('type') is None:
            date = datetime.strptime(rec['formatted_date'], '%Y-%m-%d')
            dic_with_prices(prices, ticker, date, rec['open'], rec['high'], rec['low'], rec['close'], rec['volume'])

    frame = pd.DataFrame.from_dict(prices, orient='index', columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    save_csv(base_dir, ticker, frame, 'yahoo')


# ---------------------------------------------------------------------------------------------------------------------
# Определяем стартовые даты
def date_search(f_date: datetime, *args: datetime) -> datetime:
    newest_date = f_date
    for arg in args:
        if arg > newest_date:
            newest_date = arg

    return newest_date


# Обрезаем файлы согласно определённым датам
def correct_file_by_dates(file: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return file.loc[(file['Date'] >= start) & (file['Date'] <= end)]


# Считаем VIX_Ratio
def vix_ratio(vix: pd.DataFrame, vxv: pd.DataFrame) -> list:
    temp = []
    for i in range(len(vix)):
        if style == 'open':
            temp.append(vix["Open"].iloc[i]/vxv["Open"].iloc[i])
        elif style == 'close':
            temp.append(vix["Close"].iloc[i] / vxv["Close"].iloc[i])
        else:
            print(r'Введено неверное значение "style"')
    return temp


# Считаем АТР
def atr(file: pd.DataFrame) -> list:
    temp = []
    for i in range(len(file)):
        if i >= 10 and len(temp) >= 10:
            cur_ticker = file.head(i)
            cur_ticker = cur_ticker.tail(10)
            temp.append(np.average(cur_ticker["High"] - cur_ticker["Low"].tolist()))
        else:
            temp.append(0)

    return temp


# Считаем CAGR
def cagr(file:pd.DataFrame, capital:list) -> float:
    years = (file["Date"].iloc[-1].year + file["Date"].iloc[-1].month / 12) - \
            (file["Date"].iloc[0].year + file["Date"].iloc[-1].month / 12)

    return ((capital[-1] / capital[0]) ** (1 / years) - 1) * 100


# Считаем годовое отклонение
def st_dev(capital: list) -> float:
    day_cng = []
    for i in range(len(capital)):
        if i == 0:
            day_cng.append(0)
        else:
            day_cng.append(capital[i] / capital[i - 1] - 1)
    return stat.stdev(day_cng) * math.sqrt(252)


# Считаем предельную просадку
def draw_down(capital: list) -> float:
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * 100)
    return min(down)


# Считаем сделки
def trade_count(file: pd.DataFrame, direct: int, stop: float, enter: list, stop_dinamic: bool) -> list:
    capital = []
    comm = []
    shares = []
    non_splt_shares = []

    for i in range(len(file)):
        if direct == 1 and style == 'open':
            cur_open = file['Open'][i]
            nonsplt_open = file['NonSpl_O'][i]
            stop = stop if stop_dinamic is False else file['Stop'][i]

            # Если самая первая строчка
            if i == 0:
                comm.append(0)
                non_splt_shares.append(0)
                capital.append(start_cap)
                shares.append(0)
            # Если вчера ещё были в позиции, а сегодня надо выходить
            elif enter[i] == 0 and enter[i - 1] == 1:
                comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
                non_splt_shares.append(0)
                capital.append(shares[-1] * cur_open - comm[-1])
                shares.append(0)
            # Если вчера и сегодня нужно быть вне позиции
            elif enter[i] == 0 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                comm.append(0)
                non_splt_shares.append(0)
                capital.append(capital[-1])
                shares.append(0)
            # Если получили стоп, но зашли в позу только сегодня или вчера получили стоп
            elif enter[i] == -1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                comm.append(capital[-1] / nonsplt_open * (COMM * 2 + SLIPP) if capital[-1] / nonsplt_open >= 100
                            else CONST_COMM * 2 + capital[-1] / nonsplt_open * SLIPP)
                non_splt_shares.append(0)
                capital.append(capital[-1] / cur_open * (cur_open - stop * file['ATR'][i]) - comm[-1])
                shares.append(0)
            # Если получили стоп, но были в позе ранее
            elif enter[i] == -1 and enter[i - 1] == 1:
                comm.append(non_splt_shares[-1] * (COMM + SLIPP) if non_splt_shares[-1] >= 100 else
                            CONST_COMM + non_splt_shares[-1] * SLIPP)
                non_splt_shares.append(0)
                capital.append(shares[-1] * (cur_open - stop * file['ATR'][i]) - comm[-1])
                shares.append(0)
            # Если вчера были вне позици, а сегодня надо входить
            elif enter[i] == 1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                non_splt_shares.append(capital[-1] / nonsplt_open)
                comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
                capital.append(capital[-1] - comm[-1])
                shares.append(capital[-1] / cur_open)
            # Если сегодня спокойно сидим в позиции
            elif enter[i] == 1:
                comm.append(0)
                non_splt_shares.append(non_splt_shares[-1])
                capital.append(shares[-1] * cur_open)
                shares.append(shares[-1])

    return capital


# Создаём файлы с разметками входов
def make_enters_file(file: pd.DataFrame, ticker: str, direct: int):
    for ratio in mit.numeric_range(0.88, 1.05, 0.005):
        for stop in mit.numeric_range(0.1, 5, 0.02):
            ratio = round(ratio, 3)
            stop = round(stop, 2)
            print(ratio)
            print(stop)

            if direct == 1 and style == 'open':
                enter =[]

                for i in range(len(file)):
                    if ratio < file['Open_R'][i] or file['ATR'][i] == 0:
                        enter.append(0)
                    elif ratio > file['Open_R'][i] and file["Open"][i] - stop * file['ATR'][i] >= file["Low"][i]:
                        enter.append(-1)
                    else:
                        enter.append(1)

                file['R'+str(ratio)+' S'+str(stop)] = enter

            else:
                print('Необработанный тип входа')

    save_csv(default_data_dir, str(ticker) + ' AllEnters_'+str(style), file, 'new_file')


# Перебирает файл по годам и обращается к форвардному тестеру
def years_iterator(file: pd.DataFrame, ticker: str, direct: int):
    start_year = file['Date'].iloc[0].year
    working_date = str_dt(str(start_year) + '-01-01')

    while working_date.year < datetime.now().year:
        print(working_date)
        cut_file = file.loc[file['Date'] < working_date + rd(years=1)]
        forward_files(cut_file, direct)
        working_date = working_date + rd(years=1)

    forward_table = pd.DataFrame({"Year": Year,
                                  "BestRatio": BestRatio,
                                  "BestStop": BestStop,
                                  },
                                 columns=["Year", "BestRatio", "BestStop"]
                                 )

    save_csv(default_data_dir, str(ticker) + ' BestForwards_' + str(style), forward_table, 'new_file')


# Форвардный тест и генерация файлов по годам
def forward_files(file: pd.DataFrame, direct: int):
    Ratio = []
    Stop = []
    CAGR = []
    StDev = []
    DrawDown = []
    Sharpe = []
    MaR = []
    SM = []

    for ratio in mit.numeric_range(0.88, 1.05, 0.005):
        for stop in mit.numeric_range(0.1, 5, 0.02):
            ratio, stop = round(ratio, 3), round(stop, 2)
            print(ratio, stop)

            # capital = []
            # comm = []
            # shares = []
            # non_splt_shares = []

            enter = file['R' + str(ratio) + ' S' + str(stop)]
            capital = trade_count(file, direct, stop, enter, False)
            # for i in range(len(file)):
            #     if direct == 1 and style == 'open':
            #         cur_open = file['Open'][i]
            #         nonsplt_open = file['NonSpl_O'][i]
            #
            #         # Если самая первая строчка
            #         if i == 0:
            #             comm.append(0)
            #             non_splt_shares.append(0)
            #             capital.append(start_cap)
            #             shares.append(0)
            #         # Если вчера ещё были в позиции, а сегодня надо выходить
            #         elif enter[i] == 0 and enter[i - 1] == 1:
            #             comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
            #             non_splt_shares.append(0)
            #             capital.append(shares[-1] * cur_open - comm[-1])
            #             shares.append(0)
            #         # Если вчера и сегодня нужно быть вне позиции
            #         elif enter[i] == 0 and (enter[i - 1] == 0 or enter[i - 1] == -1):
            #             comm.append(0)
            #             non_splt_shares.append(0)
            #             capital.append(capital[-1])
            #             shares.append(0)
            #         # Если получили стоп, но зашли в позу только сегодня или вчера получили стоп
            #         elif enter[i] == -1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
            #             comm.append(capital[-1] / nonsplt_open * (COMM * 2 + SLIPP) if capital[-1] / nonsplt_open >= 100
            #                         else CONST_COMM * 2 + capital[-1] / nonsplt_open * SLIPP)
            #             non_splt_shares.append(0)
            #             capital.append(capital[-1] / cur_open * (cur_open - stop * file['ATR'][i]) - comm[-1])
            #             shares.append(0)
            #         # Если получили стоп, но были в позе ранее
            #         elif enter[i] == -1 and enter[i - 1] == 1:
            #             comm.append(non_splt_shares[-1] * (COMM + SLIPP) if non_splt_shares[-1] >= 100 else
            #                         CONST_COMM + non_splt_shares[-1] * SLIPP)
            #             non_splt_shares.append(0)
            #             capital.append(shares[-1] * (cur_open - stop * file['ATR'][i]) - comm[-1])
            #             shares.append(0)
            #         # Если вчера были вне позици, а сегодня надо входить
            #         elif enter[i] == 1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
            #             non_splt_shares.append(capital[-1] / nonsplt_open)
            #             comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
            #             capital.append(capital[-1] - comm[-1])
            #             shares.append(capital[-1] / cur_open)
            #         # Если сегодня спокойно сидим в позиции
            #         elif enter[i] == 1:
            #             comm.append(0)
            #             non_splt_shares.append(non_splt_shares[-1])
            #             capital.append(shares[-1] * cur_open)
            #             shares.append(shares[-1])

            Ratio.append(ratio)
            Stop.append(stop)
            CAGR.append((capital[-1] / capital[0] - 1) * 100)
            StDev.append(st_dev(capital))
            DrawDown.append(draw_down(capital))
            Sharpe.append(CAGR[-1] / StDev[-1])
            MaR.append(abs(CAGR[-1] / DrawDown[-1]))
            SM.append(Sharpe[-1] * MaR[-1])

    temp_table = pd.DataFrame({"Ratio": Ratio,
                               "Stop": Stop,
                               "CAGR": CAGR,
                               "StDev": StDev,
                               "DrawDown": DrawDown,
                               "Sharpe": Sharpe,
                               "MaR": MaR,
                               "SM": SM},
                              columns=["Ratio", "Stop", "CAGR", "StDev", "DrawDown", "Sharpe", "MaR", "SM"]
                              )
    temp_table = temp_table.loc[temp_table['DrawDown'] >= -30.0].sort_values(by='CAGR', ascending=False)
    temp_table = temp_table.reset_index(drop=True)
    Year.append(file['Date'].iloc[-1].year + 1)
    BestRatio.append(temp_table['Ratio'][0])
    BestStop.append(temp_table['Stop'][0])

    print(len(file))
    print(Year, BestRatio, BestStop)


# Старая исполняемая функция
def total_func(file: pd.DataFrame, file_nonsplit: pd.DataFrame, ticker: str):
    for ratio in mit.numeric_range(0.88, 1.05, 0.005):
        for stop in mit.numeric_range(0.1, 5, 0.02):
            # Листы текущего блока расчётов
            enter = []
            capital = []
            comm = []
            shares = []
            non_splt_shares = []

            ratio = round(ratio, 3)
            stop = round(stop, 2)

            for i in range(len(file['Open_R'])):
                if ratio < file['Open_R'][i] or file['ATR'][i] == 0:
                    enter.append(0)
                elif ratio > file['Open_R'][i] and file["Open"][i] - stop * file['ATR'][i] >= file["Low"][i]:
                    enter.append(-1)
                else:
                    enter.append(1)

            # Блок расчёта капитала
            for i in range(len(enter)):
                cur_open = file['Open'][i]
                nonsplt_open = file_nonsplit['Open'][i]

                # Если самая первая строчка
                if i == 0:
                    comm.append(0)
                    non_splt_shares.append(0)
                    capital.append(start_cap)
                    shares.append(0)
                # Если вчера ещё были в позиции, а сегодня надо выходить
                elif enter[i] == 0 and enter[i - 1] == 1:
                    comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
                    non_splt_shares.append(0)
                    capital.append(shares[-1] * cur_open - comm[-1])
                    shares.append(0)
                # Если вчера и сегодня нужно быть вне позиции
                elif enter[i] == 0 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                    comm.append(0)
                    non_splt_shares.append(0)
                    capital.append(capital[-1])
                    shares.append(0)
                # Если получили стоп, но зашли в позу только сегодня или вчера получили стоп
                elif enter[i] == -1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                    comm.append(capital[-1] / nonsplt_open * (COMM * 2 + SLIPP) if capital[-1] / nonsplt_open >= 100
                                else CONST_COMM * 2 + capital[-1] / nonsplt_open * SLIPP)
                    non_splt_shares.append(0)
                    capital.append(capital[-1] / cur_open * (cur_open - stop * file['ATR'][i]) - comm[-1])
                    shares.append(0)
                # Если получили стоп, но были в позе ранее
                elif enter[i] == -1 and enter[i - 1] == 1:
                    comm.append(non_splt_shares[-1] * (COMM + SLIPP) if non_splt_shares[-1] >= 100 else
                                CONST_COMM + non_splt_shares[-1] * SLIPP)
                    non_splt_shares.append(0)
                    capital.append(shares[-1] * (cur_open - stop * file['ATR'][i]) - comm[-1])
                    shares.append(0)
                # Если вчера были вне позици, а сегодня надо входить
                elif enter[i] == 1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                    non_splt_shares.append(capital[-1] / nonsplt_open)
                    comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
                    capital.append(capital[-1] - comm[-1])
                    shares.append(capital[-1] / cur_open)
                # Если сегодня спокойно сидим в позиции
                elif enter[i] == 1:
                    comm.append(0)
                    non_splt_shares.append(non_splt_shares[-1])
                    capital.append(shares[-1] * cur_open)
                    shares.append(shares[-1])

            ''''# Проверка результатов
            file['Non_Price'] = file_nonsplit['Open']
            file['Enter'] = enter
            file['Comm'] = comm
            file['non_slpt_shares'] = non_splt_shares
            file['Shares'] = shares
            file['Capital'] = capital

            save_csv(default_data_dir, str(ticker) + ' Temp_File', file, 'new_file')
            '''

            day_cng = []
            for i in range(len(capital)):
                if i == 0:
                    day_cng.append(0)
                else:
                    day_cng.append(capital[i] / capital[i - 1] - 1)

            high = 0
            down = []
            for i in range(len(capital)):
                if capital[i] > high:
                    high = capital[i]
                down.append((capital[i] / high - 1) * 100)

            years = (file["Date"].iloc[-1].year + file["Date"].iloc[-1].month / 13) - \
                    (file["Date"].iloc[0].year + file["Date"].iloc[-1].month / 13)

            Ratio.append(ratio)
            Stop.append(stop)
            CAGR.append(((capital[-1] / capital[0]) ** (1 / years) - 1) * 100)
            StDev.append(stat.stdev(day_cng) * math.sqrt(252))
            DrawDown.append(min(down))
            Sharpe.append(CAGR[-1] / StDev[-1])
            MaR.append(abs(CAGR[-1] / DrawDown[-1]))
            SM.append(Sharpe[-1] * MaR[-1])

            print(ratio)
            print(stop)

    export_table = pd.DataFrame({"Ratio": Ratio,
                                "Stop": Stop,
                                "CAGR": CAGR,
                                "StDev": StDev,
                                "DrawDown": DrawDown,
                                "Sharpe": Sharpe,
                                "MaR": MaR,
                                "SM": SM},
                               columns=["Ratio", "Stop", "CAGR", "StDev", "DrawDown", "Sharpe", "MaR", "SM"]
                               )

    save_csv(default_data_dir, str(ticker) + ' Open_Metrics', export_table, 'new_file')


# Пока, чтобы не переделывать код, оставим как есть. В будущем, большое количество тикеров можно загнать в словари
'''
4) Докачка, если финальные дни не соотвествуют текущей дате
5) Дорасчёт новых строчек в уже посчитанных файлах
'''


if __name__ == '__main__':
    # Загружены ли данные по тикерам, для работы
    # download_yahoo('^VIX')
    # download_yahoo('^VXV')
    for f in analysis_tickers:
        if os.path.isfile(os.path.join(default_data_dir, str(f) + '.csv')) is False or download_data:
            download_yahoo(f)
        if os.path.isfile(os.path.join(default_data_dir, str(f) + ' NonSplit.csv')) is False or download_data:
            download_alpha(f)

    for t in range(len(analysis_tickers)):
        # Создаём файл со всеми вариантами входов, если он не создан
        if os.path.isfile(os.path.join(default_data_dir, str(analysis_tickers[t]) + ' AllEnters_' + str(style) +
                                                         '.csv')) is False:

            ticker_base = load_csv(str(analysis_tickers[t]))
            nonsplit_base = load_csv(str(analysis_tickers[t]) + ' NonSplit')
            vix_base = load_csv('VIX')
            vxv_base = load_csv('VXV')

            direct = 1 if ticker_base['Close'].iloc[-1] > ticker_base['Close'].iloc[0] else -1

            start, end = date_search(ticker_base['Date'].iloc[0], vxv_base['Date'].iloc[0]), \
                         ticker_base['Date'].iloc[-1]
            ticker_base = correct_file_by_dates(ticker_base, start, end)
            nonsplit_base = correct_file_by_dates(nonsplit_base, start, end)
            vix_base = correct_file_by_dates(vix_base, start, end)
            vxv_base = correct_file_by_dates(vxv_base, start, end)

            ticker_base['NonSpl_O'] = nonsplit_base['Open']
            ticker_base['NonSpl_C'] = nonsplit_base['Close']
            ticker_base['Open_R'] = vix_ratio(vix_base, vxv_base)
            ticker_base['ATR'] = atr(ticker_base)

            make_enters_file(ticker_base, analysis_tickers[t], direct)

        # Запускаем генератор форвардных файлов, если файл не создан
        if os.path.isfile(os.path.join(default_data_dir, str(analysis_tickers[t]) + ' BestForwards_' + str(style) +
                                                         '.csv')) is False:
            ticker_base = load_csv(str(analysis_tickers[t]) + ' AllEnters_' + str(style))
            direct = 1 if ticker_base['Close'].iloc[-1] > ticker_base['Close'].iloc[0] else -1
            years_iterator(ticker_base, analysis_tickers[t], direct)

        # Создаём файл с динамикой капитала, на основании форвард-файла
        ticker_base = load_csv(str(analysis_tickers[t]) + ' AllEnters_' + str(style))
        ratio_base = load_csv(str(analysis_tickers[t]) + ' BestForwards_' + str(style)).set_index(['Year'])


        ticker_base = ticker_base.loc[
            ticker_base['Date'] >= str_dt(str(ticker_base['Date'].iloc[0].year + 1) + '-01-01')].reset_index(drop=True)

        temp1 = []
        temp2 = []
        temp3 = []
        for i in range(len(ticker_base)):
            cur_year = ticker_base['Date'].iloc[i].year
            temp1.append(ratio_base['BestRatio'][cur_year])
            temp2.append(ratio_base['BestStop'][cur_year])
            if i == 0:
                temp3.append(0)
            else:
                temp3.append(ticker_base['R' + str(temp1[-1]) + ' S' + str(temp2[-1])][i])

        ticker_base['Ratio'] = temp1
        ticker_base['Stop'] = temp2
        ticker_base['Enter'] = temp3

        print(ticker_base)

        true_columns = []
        for column in ticker_base:
            if 'S' in column and column[:1] == 'R':
                continue
            else:
                true_columns.append(column)
        ticker_base = ticker_base.reindex(columns=true_columns).reset_index(drop=True)

        with pd.option_context('display.max_columns', 20):
            print(ticker_base)

        direct = 1 if ticker_base['Close'].iloc[-1] > ticker_base['Close'].iloc[0] else -1
        capital = trade_count(ticker_base, direct, 0, ticker_base['Enter'], True)

        wr = csv.writer(open('myfile.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(capital)