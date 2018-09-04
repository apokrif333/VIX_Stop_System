import pandas as pd
import numpy as np
import more_itertools as mit
import statistics as stat
import matplotlib.pyplot as plt
import math
import time
import cmath
import os

from alpha_vantage.timeseries import TimeSeries
from yahoofinancials import YahooFinancials
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd

# Const
ALPHA_KEY = 'FE8STYV4I7XHRIAI'
COMM = 0.0055  # Коэффициент размещения денег в одну сторону на 40$ цены тикера
CONST_COMM = 0.55  # Статичная комиссия, если объём меньше 100 акций
SLIPP = 0.02  # При каждом стопе проскальзывание 2 цента на акцию
R_START, R_END = 0.88, 1.1
S_START, S_END = 0.1, 5

# Variables
analysis_tickers = ['TLT']  # Чтобы скачать с yahoo, нужно выставить время в компьютере NY
start_cap = 10000

default_data_dir = 'exportTables'  # Директория
download_data = False  # Качает сплитованные данные с yahoo и не сплитованные с alpha. На alpha задержка 15 сек
start_date = datetime(1990, 1, 1)  # Для yahoo, alpha выкачает всю доступную историю
end_date = datetime.now()

style = 'open'.lower()  # open - выстраивает логику с открытия сессии, close, выстраивает логику на закрытие
ratio_step = 0.005
stop_step = 0.1
forward_analyse = False  # Создавать ли форвард-файлы с метриками по годам
file3D = False  # Создавать ли файл для 3D модели
user_enter = False  # Указывать ли вручную метрики для построения финальной таблицы

# Globals
alpha_count = 0


# Формат даты в сторку
def dt_str(date: datetime) -> str:
    return "%04d-%02d-%02d" % (date.year, date.month, date.day)


# Формат строки в дату
def str_dt(date: str) -> datetime:
    return datetime.strptime(date, '%Y-%m-%d')


# Формат листа со строками в лист с датами
def st_date(file: pd.DataFrame):
    try:
        file["Date"] = pd.to_datetime(file["Date"], dayfirst=False)
    except:
        file["Date"] = pd.to_datetime(file["Date"], format='%d-%m-%Y')


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
def save_csv(base_dir: str, file_name: str, data: pd.DataFrame, source: str):
    path = os.path.join(base_dir)
    if not os.path.exists(path):
        os.makedirs(path)

    if source == 'alpha':
        print(f'{file_name} работает с альфой')
        path = os.path.join(path, file_name + ' NonSplit' + '.csv')
    elif source == 'yahoo':
        print(f'{file_name} работает с яху')
        path = os.path.join(path, file_name + '.csv')
        path = path.replace('^', '')
    elif source == 'new_file':
        print(f'Сохраняем файл с тикером {file_name}')
        path = os.path.join(path, file_name + '.csv')
    else:
        print(f'Неопознанный источник данных для {file_name}')

    if source == ('alpha' or 'yahoo'):
        data.to_csv(path, index_label='Date')
    else:
        data.to_csv(path, index=False)


# Загружаем csv файл
def load_csv(ticker: str, base_dir: str=default_data_dir) -> pd.DataFrame:
    path = os.path.join(base_dir, str(ticker) + '.csv')
    file = pd.read_csv(path)
    if 'Date' in file.columns:
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
def cagr(file: pd.DataFrame, capital: list) -> float:
    years = (file["Date"].iloc[-1].year + file["Date"].iloc[-1].month / 12) - \
            (file["Date"].iloc[0].year + file["Date"].iloc[0].month / 12)
    return ((capital[-1] / capital[0]) ** (1 / years) - 1) * 100


# Считаем годовое отклонение
def st_dev(capital: list) -> float:
    day_cng = []
    for i in range(len(capital)):
        if i == 0:
            day_cng.append(0)
        else:
            day_cng.append(capital[i] / capital[i - 1] - 1)
    return stat.stdev(day_cng) * math.sqrt(252) if stat.stdev(day_cng) != 0 else 999


# Считаем предельную просадку
def draw_down(capital: list) -> float:
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * 100)
    return min(down) if min(down) != 0 else 1


# Считаем сделки
def trade_count(file: pd.DataFrame, direct: int, stop: float, enter: list, stop_dinamic: bool) -> list:
    capital = []
    comm = []
    shares = []
    non_splt_shares = []
    prev_enter = []

    for i in range(len(file)):
        stop = stop if stop_dinamic is False else file['Stop'][i]

        if direct == 1 and style == 'open':
            cur_open = file['Open'][i]
            nonsplt_open = file['NonSpl_O'][i]

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

        elif direct == -1 and style == 'open':
            cur_open = file['Open'][i]
            nonsplt_open = file['NonSpl_O'][i]

            # Если самая первая строчка
            if i == 0:
                comm.append(0)
                non_splt_shares.append(0)
                capital.append(start_cap)
                prev_enter.append(0)
                shares.append(0)
            # Если вчера ещё были в позиции, а сегодня надо выходить
            elif enter[i] == 0 and enter[i - 1] == 1:
                comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
                non_splt_shares.append(0)
                capital.append(shares[-1] * prev_enter[-1] + (prev_enter[-1] - cur_open)  * shares[-1] - comm[-1])
                prev_enter.append(0)
                shares.append(0)
            # Если вчера и сегодня нужно быть вне позиции
            elif enter[i] == 0 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                comm.append(0)
                non_splt_shares.append(0)
                capital.append(capital[-1])
                prev_enter.append(0)
                shares.append(0)
            # Если получили стоп, но зашли в позу только сегодня или вчера получили стоп
            elif enter[i] == -1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                comm.append(capital[-1] / nonsplt_open * (COMM * 2 + SLIPP) if capital[-1] / nonsplt_open >= 100
                            else CONST_COMM * 2 + capital[-1] / nonsplt_open * SLIPP)
                non_splt_shares.append(0)
                capital.append(
                    capital[-1] + capital[-1] / cur_open * (cur_open - (cur_open + stop * file['ATR'][i])) - comm[-1])
                prev_enter.append(0)
                shares.append(0)
            # Если получили стоп, но были в позе ранее
            elif enter[i] == -1 and enter[i - 1] == 1:
                comm.append(non_splt_shares[-1] * (COMM + SLIPP) if non_splt_shares[-1] >= 100 else
                            CONST_COMM + non_splt_shares[-1] * SLIPP)
                non_splt_shares.append(0)
                capital.append(
                    shares[-1] * prev_enter[-1] + shares[-1] * (prev_enter[-1] - (cur_open + stop * file['ATR'][i])) -
                    comm[-1])
                prev_enter.append(0)
                shares.append(0)
            # Если вчера были вне позици, а сегодня надо входить
            elif enter[i] == 1 and (enter[i - 1] == 0 or enter[i - 1] == -1):
                non_splt_shares.append(capital[-1] / nonsplt_open)
                comm.append(non_splt_shares[-1] * COMM if non_splt_shares[-1] >= 100 else CONST_COMM)
                capital.append(capital[-1] - comm[-1])
                prev_enter.append(cur_open)
                shares.append(capital[-1] / cur_open)
            # Если сегодня спокойно сидим в позиции
            elif enter[i] == 1:
                comm.append(0)
                non_splt_shares.append(non_splt_shares[-1])
                capital.append(shares[-1] * prev_enter[-1] + shares[-1] * (prev_enter[-1] - cur_open))
                prev_enter.append(prev_enter[-1])
                shares.append(shares[-1])

        elif direct == 1 and style == 'close':
            pass

        elif direct == -1 and style == 'close':
            pass

        else:
            print(f'Передан или неверный директ для сделок - {direct}, или неверный style - {style}')


    return capital


# Создаём файлы с разметками входов
def make_enters_file(file: pd.DataFrame, ticker: str, direct: int):
    for ratio in mit.numeric_range(R_START, R_END, ratio_step):
        for stop in mit.numeric_range(S_START, S_END, stop_step):
            ratio = round(ratio, 3)
            stop = round(stop, 2)
            print(ratio)
            print(stop)

            enter = []
            if direct == 1 and style == 'open':
                for i in range(len(file)):
                    if ratio < file['Open_R'][i] or file['ATR'][i] == 0:
                        enter.append(0)
                    elif ratio > file['Open_R'][i] and file["Open"][i] - stop * file['ATR'][i] >= file["Low"][i]:
                        enter.append(-1)
                    else:
                        enter.append(1)

            elif direct == -1 and style == 'open':
                for i in range(len(file)):
                    if ratio < file['Open_R'][i] or file['ATR'][i] == 0:
                        enter.append(0)
                    elif ratio > file['Open_R'][i] and file["Open"][i] + stop * file['ATR'][i] <= file['High'][i]:
                        enter.append(-1)
                    else:
                        enter.append(1)

            else:
                print('Необработанный тип входа')
            file['R' + str(ratio) + ' S' + str(stop)] = enter

    save_csv(default_data_dir, str(ticker) + ' AllEnters_'+str(style), file, 'new_file')


# Перебирает файл по годам и обращается к форвардному тестеру
def years_iterator(file: pd.DataFrame, ticker: str, direct: int):
    start_year = file['Date'].iloc[0].year
    working_date = str_dt(str(start_year) + '-01-01')

    while working_date.year < datetime.now().year + 1:
        print(working_date)
        cut_file = file.loc[file['Date'] < working_date + rd(years=1)]
        forward_files(cut_file, ticker, direct)
        working_date = working_date + rd(years=1)


# Форвардный тест и генерация файлов по годам
def forward_files(file: pd.DataFrame, ticker: str, direct: int):
    Ratio = []
    Stop = []
    CAGR = []
    StDev = []
    DrawDown = []
    Sharpe = []
    MaR = []
    SM = []

    for ratio in mit.numeric_range(R_START, R_END, ratio_step):
        for stop in mit.numeric_range(S_START, S_END, stop_step):
            ratio, stop = round(ratio, 3), round(stop, 2)
            print(ratio, stop)
            enter = file['R' + str(ratio) + ' S' + str(stop)]
            capital = trade_count(file, direct, stop, enter, False)

            Ratio.append(ratio)
            Stop.append(stop)
            CAGR.append(cagr(file, capital))
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
    temp_table = temp_table.sort_values(by='SM',ascending=False).reset_index(drop=True)
    year_for_use = file['Date'].iloc[-1].year + 1

    save_csv(default_data_dir + '/temp', str(ticker) + ' _' + str(style) + '_metricFor ' + str(year_for_use),
             temp_table, 'new_file')

    print(len(file), year_for_use)


# Создаём файл для 3D модели
def model_3D(file: pd.DataFrame, ticker: str):
    new_base = pd.DataFrame({})
    new_base[''] = sorted(file['Ratio'].unique())

    stop_list = sorted(file['Stop'].unique())
    for i in stop_list:
        new_base[str(i)] = 0

    for i in range(len(new_base)):
        for s in stop_list:
            new_base.loc[i, str(s)] = file['SM'].loc[
                (file['Ratio'] == new_base[''][i]) & (file['Stop'] == s)].item()

    save_csv(default_data_dir + '/temp', ticker + ' _3D 2019', new_base, 'new_file')


# Создаём словарь с пользовательскими метриками по годам
def years_dict(file: pd.DataFrame, ticker: str) -> dict:
    temp = {}
    for i in range(len(file)):
        cur_year = file['Date'].iloc[i].year
        if str(cur_year) not in temp.keys():
            if user_enter:
                print(f'Для {ticker} введите Ratio и Stop для {cur_year} года через пробел')
                temp[str(cur_year)] = [float(_) for _ in input().split()]
            else:
                temp[str(cur_year)] = [0.94, 0.5]
    return temp


# Рисуем график с данными
def plot_chart(file: pd.DataFrame, capital: list, years_dict: dict):
    high = 0
    down = []
    for i in range(len(capital)):
        if capital[i] > high:
            high = capital[i]
        down.append((capital[i] / high - 1) * -100)

    names = ['Start Balance', 'End Balance', 'CAGR', 'DrawDown', 'StDev', 'Sharpe', 'MaR', 'SM']
    values = []
    values.append(capital[0])
    values.append(round(capital[-1],0))
    values.append(round(cagr(file, capital), 2))
    values.append(round(draw_down(capital), 2))
    values.append(round(st_dev(capital) * 100, 2))
    values.append(round(values[2] / values[4], 2))
    values.append(round(abs(values[2] / values[3]), 2))
    values.append(round(values[5] * values[6], 2))

    for key in years_dict.keys():
        names.append(key)
        values.append(years_dict[key])
    while len(names) % 4 != 0:
        names.append('')
        values.append('')

    table_metric = pd.DataFrame({'h1': names[:int(len(names) / 4)],
                                 'v1': values[:int(len(values) / 4)],
                                 'h2': names[int(len(names) / 4):int(len(names) / 2)],
                                 'v2': values[int(len(values) / 4):int(len(values) / 2)],
                                 'h3': names[int(len(names) / 2):int(len(names) * 0.75)],
                                 'v3': values[int(len(values) / 2):int(len(values) * 0.75)],
                                 'h4': names[int(len(names) * 0.75):len(names)],
                                 'v4': values[int(len(values) * 0.75):len(values)],
                                 })

    fig = plt.figure(figsize=(12.8, 8.6), dpi=80)

    ax1 = fig.add_subplot(6, 1, (1, 5))
    ax1.plot(file['Date'], down, dashes=[6, 4], color="darkgreen", alpha=0.5)
    ax1.set_ylabel('Просадки')

    ax2 = ax1.twinx()
    ax2.plot(file['Date'], capital)
    ax2.set_ylabel('Динамика капитала')

    tx1 = fig.add_subplot(6, 1, 6, frameon=False)
    tx1.axis("off")
    tx1.table(cellText=table_metric.values, loc='lower center')

    plt.show()


# Пока, чтобы не переделывать код, оставим как есть. В будущем, большое количество тикеров можно загнать в словари
'''
4) Докачка, если финальные дни не соотвествуют текущей дате
5) Дорасчёт новых строчек в уже посчитанных файлах
'''

if __name__ == '__main__':
    # Блок загрузки данных
    for f in analysis_tickers:
        if os.path.isfile(os.path.join(default_data_dir, str(f) + '.csv')) is False or download_data:
            download_yahoo(f)
        if os.path.isfile(os.path.join(default_data_dir, str(f) + ' NonSplit.csv')) is False or download_data:
            download_alpha(f)

    # Основной рабочий блок
    for t in range(len(analysis_tickers)):
        # Создаём файл со всеми вариантами входов, если он не создан
        if os.path.isfile(os.path.join(default_data_dir,
                                       str(analysis_tickers[t]) + ' AllEnters_' + str(style) + '.csv')) is False:
            ticker_base = load_csv(str(analysis_tickers[t]))
            nonsplit_base = load_csv(str(analysis_tickers[t]) + ' NonSplit')
            vix_base = load_csv('VIX')
            vxv_base = load_csv('VXV')

            direct = 1 if ticker_base['Close'].iloc[-1] > ticker_base['Close'].iloc[0] else -1

            start, end = date_search(ticker_base['Date'].iloc[0], vxv_base['Date'].iloc[0]), ticker_base['Date'].iloc[
                -1]
            ticker_base = correct_file_by_dates(ticker_base, start, end)
            nonsplit_base = correct_file_by_dates(nonsplit_base, start, end)
            vix_base = correct_file_by_dates(vix_base, start, end)
            vxv_base = correct_file_by_dates(vxv_base, start, end)

            ticker_base['NonSpl_O'] = nonsplit_base['Open']
            ticker_base['NonSpl_C'] = nonsplit_base['Close']
            ticker_base['Open_R'] = vix_ratio(vix_base, vxv_base)
            ticker_base['ATR'] = atr(ticker_base)

            ticker_base = ticker_base.reset_index(drop=True)
            make_enters_file(ticker_base, analysis_tickers[t], direct)

        # Запускаем генератор форвардных файлов выдающий анализ всех переменных по годам
        if forward_analyse:
            ticker_base = load_csv(str(analysis_tickers[t]) + ' AllEnters_' + str(style))
            direct = 1 if ticker_base['Close'].iloc[-1] > ticker_base['Close'].iloc[0] else -1
            years_iterator(ticker_base, analysis_tickers[t], direct)

        # Создаём файл для отрисовки 3D-модели
        if file3D:
            ticker_base = load_csv(analysis_tickers[t] + ' _open_metricFor 2019', default_data_dir + '/temp')
            model_3D(ticker_base, analysis_tickers[t])

        # Выводим plot капитала, на основе указанных переменных для разных лет
        ticker_base = load_csv(str(analysis_tickers[t]) + ' AllEnters_' + str(style))
        ticker_base = ticker_base.loc[
            ticker_base['Date'] >= str_dt(str(ticker_base['Date'].iloc[0].year + 1) + '-01-01')].reset_index(drop=True)
        years = years_dict(ticker_base, analysis_tickers[t])

        ratio = []
        stop = []
        enter = []
        for i in range(len(ticker_base)):
            cur_year = ticker_base['Date'].iloc[i].year
            ratio.append(years[str(cur_year)][0])
            stop.append(years[str(cur_year)][1])
            if i == 0:
                enter.append(0)
            else:
                enter.append(ticker_base['R' + str(ratio[-1]) + ' S' + str(stop[-1])][i])
        ticker_base['Ratio'] = ratio
        ticker_base['Stop'] = stop
        ticker_base['Enter'] = enter

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

        plot_chart(ticker_base, capital, years)

        # final_table = pd.DataFrame({'Date': ticker_base['Date'], 'Capital': capital})
        # save_csv(default_data_dir, analysis_tickers[t] + ' _finalCapital', final_table, 'new_file')