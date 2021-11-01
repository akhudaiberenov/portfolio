import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import pytz
import matplotlib.pyplot as plt
import mplfinance as mpf

import os
from os import walk
import numpy as np
import time as time_

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class range_detection:

    def __init__(self):

        self.list_atr_appropriate_symbols = None
        self.list_median_appropriate_symbols = None
        self.list_square_appropriate_symbols = None

        self.MIN_COUNT_BARS = 5  # минимально число баров в редже
        self.MAX_COUNT_BARS = 10  # максимальное число баров в рендже
        self.DEGREE = 1  # степень кривой полиномиальной регресиии

        # максимальная разница коэффициентов между асимптотами по хаям и лоям
        self.MAX_INCLINE_DIFF = 0.15
        self.MAX_INCLINE_DIFF_STEP = 0.001
        # ограничения по наклону и кривизны дял всех асимптот
        self.MAX_INCLINE = 0.15
        self.MAX_INCLINE_STEP = 0.002
        # минимальная заполенность ренджа
        self.MIN_FULLNESS_RANGE = 0.7
        self.MIN_FULLNESS_RANGE_KOEF = 0.015
        self.MIN_FULLNESS_RANGE_CURTAGE = 0.9
        self.COEF_ANTI_AREA_OF_RANGE = 2

        # коэффициетн максимальный высоты ренджа
        self.MAX_HEIHGT_OF_RANGE_KOEF = 0.3
        # степень кривизны графика увеличения допустимой ширины ренджа
        self.MAX_HEIHGT_CURTAGE_CHART = 0.8

        # Максимальная доля отбора максимальных и минимальных баров при смещении асимптоты
        self.MAX_PERCENT_OF_MAX_MIN_BARS = 1
        # два нижних показателя при увеличении числа баров требуют пересмотра
        # коэффициент процентного изменения
        self.DIFF_PERCENT_OF_MAX_MIN_BARS = 0.05
        # коэффициент нелинейности
        self.CURTAGE_PERCENT_OF_MAX_MIN_BARS = 0.5

        # Количество баров для отрисовки
        self.TOTAL_COUNT_PLOT_BARS = 149
        # Количество баров при расчете АТР
        self.COUNT_ATR_BARS = 300
        self.HIGH_BORDER = 3
        self.LOW_BORDER = 1 / 3
        self.LENGHT_OF_BAR_IN_ATR = 0.2  # "длина" одного бара в атр

        self.all_plot_data = None
        self.count_empty_bars = 1

    def __call__(self, write_data_to_excel=False):

        list_list_count_bars = []
        list_plot_list_high_max = []
        list_plot_list_low_min = []
        list_list_coefs_range = []
        list_list_coefs_high = []
        list_list_coefs_low = []
        list_list_fullnes_of_range = []
        list_list_appropriate_symbols = []
        list_list_plot_data = []
        list_list_diffs_high_low_incline = []

        mypath = Path("/content/drive/MyDrive/TC/data")
        _, _, self.LIST_SYMBOLS = next(walk(mypath), (None, None, []))

        for symbol in self.LIST_SYMBOLS:

            # try:
            path = "/content/drive/MyDrive/TC/data/" + symbol
            # np.seterr('raise')
            row_data = pd.read_csv(path, sep=",")
            data = self.process_data(row_data)
            data = data.iloc[-self.TOTAL_COUNT_PLOT_BARS:][::-1]
            atr_data = data.copy().iloc[-self.COUNT_ATR_BARS:]  # данные для получения атр

            diffs = atr_data.High.values - atr_data.Low.values  # получаем разницы хаев лоев
            atr = self.get_atr(diffs)  # получаем атр с учетом отброса аномалий

            # Ко всем данным прибавляем пустые значения для пустоты справа, которую будте занимать визуализация спреда
            # к датафрейму с OHLC
            utc_now = pytz.utc.localize(datetime.utcnow())
            pst_now = utc_now.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0, minute=0)
            delta = timedelta(hours=1)

            datetime_list = []
            for dt in [pst_now + delta * i for i in range(self.count_empty_bars)]:
                datetime_list.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            empty_df = pd.DataFrame([[np.nan] * len(data.columns.values) for i in range(self.count_empty_bars)],
                                    columns=data.columns.values, index=datetime_list)
            data_plot = data.copy()[::-1].append(empty_df)
            data_plot.index = pd.to_datetime(data_plot.index)
            list_list_plot_data.append(data_plot)

            # инициализирую список длин ренджей в текущем времени
            list_count_bars = []
            list_list_high_max = []
            list_list_low_min = []
            list_coefs_range = []
            list_coefs_high = []
            list_coefs_low = []
            list_fullnes_of_range = []
            list_diffs_high_low_incline = []

            plot_list_high_max = []
            plot_list_low_min = []

            for count_bars in range(self.MIN_COUNT_BARS, self.MAX_COUNT_BARS + 1):

                min_fullnes_range = self.MIN_FULLNESS_RANGE - (
                        1 + self.MIN_FULLNESS_RANGE_KOEF * count_bars) ** self.MIN_FULLNESS_RANGE_CURTAGE + 1
                max_incline = self.MAX_INCLINE
                max_incline_diff = self.MAX_INCLINE_DIFF
                percent_of_min_max_bars = self.MAX_PERCENT_OF_MAX_MIN_BARS - (
                        1 + self.DIFF_PERCENT_OF_MAX_MIN_BARS * count_bars) ** self.CURTAGE_PERCENT_OF_MAX_MIN_BARS + 1
                max_height_of_range = atr * (
                        1 + self.MAX_HEIHGT_OF_RANGE_KOEF * count_bars) ** self.MAX_HEIHGT_CURTAGE_CHART

                ## Ищем рендж методом ограничения наклона и кривизны асимптоты точек хаев и
                ## лоев вместе и разницы их асимптот хаев и лоев
                # Обрезаем данные
                data_cut = data.iloc[:count_bars].copy()
                # Получаем датафреймы с координатами точек
                coordinates_dataframe, high_low_coordinates_dataframe = self.get_points_coordinates(data_cut,
                                                                                                    count_bars, atr)

                # Получаем кортеж коэффициентов кривизны и наклона по хаям и лоям вместе
                model_range_coefs, range_asymptote = self.train_polynomial(coordinates_dataframe, "x_points",
                                                                           "y_points")
                # Получаем кортеж коэффициентов кривизны и наклона по хаям и лоям раздельно
                model_high_coefs, high_asymptote = self.train_polynomial(high_low_coordinates_dataframe, "x_points",
                                                                         "y_high_points")
                model_low_coefs, low_asymptote = self.train_polynomial(high_low_coordinates_dataframe, "x_points",
                                                                       "y_low_points")

                # относительная разница коэффициентов наклона асимптоты
                diff_high_low_incline = abs(model_high_coefs[0] - model_low_coefs[0])

                # Ищем заполненность ренджа
                # получаем, сдвинутые к максимальным смещениям, асимптоты, которые по сути, являются границами ренджа
                high_border, low_border = self.get_borders(high_low_coordinates_dataframe, high_asymptote,
                                                           low_asymptote, percent_of_min_max_bars)
                # получаем относительную заполненность ренджа
                fullness_relation = self.fullness_of_range(high_low_coordinates_dataframe, high_border, low_border)
                # высота ренджа
                height_of_range = high_border.mean() - low_border.mean()

                if abs(model_range_coefs[0]) <= max_incline and \
                        diff_high_low_incline <= max_incline_diff and \
                        model_high_coefs[0] <= max_incline and \
                        model_low_coefs[0] <= max_incline and \
                        height_of_range <= max_height_of_range and \
                        fullness_relation >= min_fullnes_range:
                    # если условия выполняются добавляем число баров в рендже в список числа баров,
                    list_count_bars.append(count_bars)
                    list_list_high_max.append(high_border.tolist())
                    list_list_low_min.append(low_border.tolist())
                    list_coefs_range.append(model_range_coefs[0])
                    list_coefs_high.append(model_high_coefs[0])
                    list_coefs_low.append(model_low_coefs[0])
                    list_fullnes_of_range.append(fullness_relation)
                    list_diffs_high_low_incline.append(diff_high_low_incline)

            if list_count_bars:
                # индекс макисмального ренджа
                index_of_max_range = list_count_bars.index(max(list_count_bars))
                max_count_range_bars = max(list_count_bars)
                # доюавляем верхнюю и нижню границу ренджа для текущего времени
                plot_list_high_max = [np.nan] * (self.TOTAL_COUNT_PLOT_BARS - max_count_range_bars) + \
                    list_list_high_max[index_of_max_range] + [np.nan] * self.count_empty_bars
                plot_list_low_min = [np.nan] * (self.TOTAL_COUNT_PLOT_BARS - max_count_range_bars) + list_list_low_min[
                    index_of_max_range] + [np.nan] * self.count_empty_bars
                # остальные данные
                coef_range = list_coefs_range[index_of_max_range]
                coef_high = list_coefs_high[index_of_max_range]
                coef_low = list_coefs_low[index_of_max_range]
                fullnes_of_range = list_fullnes_of_range[index_of_max_range]
                diff_high_low_incline = list_diffs_high_low_incline[index_of_max_range]

                list_list_appropriate_symbols.append(symbol)
                list_list_count_bars.append(max_count_range_bars)
                list_plot_list_high_max.append(plot_list_high_max)
                list_plot_list_low_min.append(plot_list_low_min)
                list_list_coefs_range.append(coef_range)
                list_list_coefs_high.append(coef_high)
                list_list_coefs_low.append(coef_low)
                list_list_fullnes_of_range.append(fullnes_of_range)
                list_list_diffs_high_low_incline.append(diff_high_low_incline)

            # except BaseException:
            #     print("Ошибка при робрботке файла: ", symbol)

        self.all_plot_data = {
            "appropriate_symbols": list_list_appropriate_symbols,
            "count_bars": list_list_count_bars,
            "plot_high_max": list_plot_list_high_max,
            "plot_low_min": list_plot_list_low_min,
            "coef_range": list_list_coefs_range,
            "coef_high": list_list_coefs_high,
            "coef_low": list_list_coefs_low,
            "diff_h_l_incline": list_list_diffs_high_low_incline,
            "fullnes_of_range": list_list_fullnes_of_range,
            "plot_data": list_list_plot_data
        }

    def do_calculations(self):
        """
        Функция выполняет основной блок кода
        """
        pass

    def plot_all_ranges(self):
        """
        Функция показывает графики для всех активов имеющих рендж
        """
        # определяем количество графиков
        list_symbols_without_range = []
        list_data_without_range = []
        list_data_with_range = []
        for symbol, data in zip(self.LIST_SYMBOLS, self.all_plot_data["plot_data"]):
            if symbol not in self.all_plot_data["appropriate_symbols"]:
                list_symbols_without_range.append(symbol)
                list_data_without_range.append(data)
            else:
                list_data_with_range.append(data)

        list_symbols_with_and_without_range = self.all_plot_data["appropriate_symbols"] + list_symbols_without_range
        data = list_data_with_range + list_data_without_range

        num_columns = 3
        if len(list_symbols_with_and_without_range) > num_columns:
            import math
            num_rows = math.ceil(len(list_symbols_with_and_without_range) / 3)
        else:
            num_rows = 1
            num_columns = len(list_symbols_with_and_without_range)

        # Визуализация
        fig = mpf.figure(figsize=(15 * num_columns, 10 * num_rows))
        axes = fig.subplots(num_rows, num_columns, sharex=False, squeeze=False)

        num = 0
        for ax in axes.ravel():
            if num < len(list_symbols_with_and_without_range):
                symbol = list_symbols_with_and_without_range[num]
                plot_data = data[num]

                # Рассчитываем спред
                bid_path = "/content/drive/MyDrive/TC/data/spread/" + symbol[:-4] + "_bid" + ".csv"
                ask_path = "/content/drive/MyDrive/TC/data/spread/" + symbol[:-4] + "_ask" + ".csv"

                row_bid_data = pd.read_csv(bid_path, sep=",")
                bid_data = self.process_data(row_bid_data, time_only=False, ignore_volume=True)
                current_bid = bid_data.iloc[-1].copy().Close

                row_ask_data = pd.read_csv(ask_path, sep=",")
                aks_data = self.process_data(row_ask_data, time_only=False, ignore_volume=True)
                current_ask = aks_data.iloc[-1].copy().Close

                spread = current_ask - current_bid
                # Данные для визуализации спреда
                bid_values = [np.nan] * (self.TOTAL_COUNT_PLOT_BARS - 1 + self.count_empty_bars)
                ask_values = [np.nan] * (self.TOTAL_COUNT_PLOT_BARS - 1 + self.count_empty_bars)
                if bid_data.Close.values[-1] >= bid_data.Open.values[-1]:
                    bid_values.append(plot_data.Close.values[-2])
                    ask_values.append(plot_data.Close.values[-2] + spread)
                else:
                    ask_values.append(plot_data.Close.values[-2])
                    bid_values.append(plot_data.Close.values[-2] - spread)

                # визуализация в зависимости от наличия ренджа в данный момент

                if num > len(self.all_plot_data["appropriate_symbols"]) - 1:
                    ax.set_title("Инструмент: " + symbol[:-4] + "     " + "СПРЭД: %.5f \n" % spread, fontsize=12,
                                 loc="left")

                    apds = [mpf.make_addplot(ask_values, type='scatter', markersize=50, marker='<', ax=ax),
                            mpf.make_addplot(bid_values, type='scatter', markersize=50, marker='<', ax=ax),
                            ]
                    mpf.plot(plot_data, type='candle', addplot=apds, ax=ax)

                else:
                    count_bars = self.all_plot_data["count_bars"][num]
                    plot_high_max = self.all_plot_data["plot_high_max"][num]
                    plot_low_min = self.all_plot_data["plot_low_min"][num]
                    coef_range = self.all_plot_data["coef_range"][num]
                    coef_high = self.all_plot_data["coef_high"][num]
                    coef_low = self.all_plot_data["coef_low"][num]
                    fullnes_of_range = self.all_plot_data["fullnes_of_range"][num]
                    diff_h_l_incline = self.all_plot_data["diff_h_l_incline"][num]

                    ax.set_title(
                        "Инструмент: " + symbol[:-4] + "     " + "Количетсво баров: " + str(
                            count_bars) + "     " + "СПРЭД: %.5f \n" % spread + \
                        "Коэффициент наклона ренджа: %.3f \n" % coef_range + \
                        "Разница наклонов асимптот хаев и лоев: %.3f \n" % diff_h_l_incline + \
                        "Коэффициент наклона хаев: %.3f \n" % coef_high + "Коэффициент наклона лоев: %.3f \n" \
                        % coef_low + "Заполненность ренджа %.3f \n" % fullnes_of_range,
                        fontsize=12, loc="left"
                    )

                    apds = [mpf.make_addplot(plot_high_max, ax=ax),
                            mpf.make_addplot(plot_low_min, ax=ax),
                            mpf.make_addplot(ask_values, type='scatter', markersize=50, marker='<', ax=ax),
                            mpf.make_addplot(bid_values, type='scatter', markersize=50, marker='<', ax=ax)
                            ]
                    mpf.plot(plot_data, type='candle', addplot=apds, ax=ax)
            else:
                pass
            num += 1

        plt.tight_layout()

        utc_now = pytz.utc.localize(datetime.utcnow())
        pst_now = utc_now.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0, minute=0)
        date = pst_now.strftime("%m_%d_%Y")
        time = pst_now.strftime("%H:%M:%S")

        if not os.path.exists("/content/drive/MyDrive/TC/data/images"):
            os.makedirs("/content/drive/MyDrive/TC/data/images")

        if not os.path.exists("/content/drive/MyDrive/TC/data/images/" + date):
            os.makedirs("/content/drive/MyDrive/TC/data/images/" + date)

        plt.savefig("/content/drive/MyDrive/TC/data/images/" + date + "/" + time + ".png")

    def write_data(self):
        """
        Функция выполняет запись данных
        """
        # Текущие дата и вермя в часах (без минут и секунд)
        utc_now = pytz.utc.localize(datetime.utcnow())
        pst_now = utc_now.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0, minute=0)

        time = pst_now.strftime("%H:%M:%S")

        self.list_datetime = [time] * len(
            self.list_median_appropriate_symbols +
            self.list_square_appropriate_symbols +
            self.list_atr_appropriate_symbols)

        # Создаем датафрейм, в котором инструменты с ренджами суммируются возмжны дубликаты)
        data_1 = {
            "time": self.list_datetime + [None],
            "symbols": self.list_symbols_with_range + [None],
            "count_bars_in_range": self.list_bars_count + [None]
        }

        self.df_1 = pd.DataFrame(data=data_1, columns=["time", "symbols", "count_bars_in_range"])
        date = pst_now.strftime("%d_%m_%Y")

        # Путь к файлу
        file_path_ = Path("/content/drive/MyDrive/TC/Suspected_assets/" + date + ".xlsx")
        file_path = "/content/drive/MyDrive/TC/Suspected_assets/" + date + ".xlsx"
        if file_path_.exists():
            # лист где были выданы все предлагаемые инструменты с ренджами в том числе и с повторениями
            self.df_01 = pd.read_excel(io=file_path, sheet_name='all_ragnes')
            self.df_01 = pd.concat([self.df_01, self.df_1])
        else:
            self.df_01 = self.df_1

        # Выполняем запись файла
        self.df_01.reset_index().to_excel(file_path, columns=["time", "symbols", "count_bars_in_range"],
                                          sheet_name='all_ragnes')

    def get_points_coordinates(self, data, count_bars, atr, high_and_low_only=True):
        """
        Функция возвращает координаты точек для определения асимптоты
        """
        # "длина" ренджа
        lengt_of_range = atr * self.LENGHT_OF_BAR_IN_ATR * count_bars
        # определяем координаты точек ренджа по горизонтали: общая длина ренджа равна его высоте
        list_horizontal_points_coordinates = np.linspace(0, lengt_of_range, count_bars, endpoint=True)

        # умножаем на 2, потому что имеес список точек high и low
        if high_and_low_only:
            list_horizontal_points_coordinates_x = list_horizontal_points_coordinates * 2
            # создаем список координат точек по вертикали
            list_vertical_points_coordinates = data.High.values.tolist() + data.Low.values.tolist()
        # если используем ohlc то умножаем на 4
        else:
            list_horizontal_points_coordinates_x = list_horizontal_points_coordinates * 4
            # создаем список координат точек по вертикали
            list_vertical_points_coordinates = \
                data.Open.values.tolist() + data.High.values.tolist() + \
                data.Low.values.tolist() + data.Close.values.tolist()

        coordinates_dataframe_dict = {
            "x_points": list_horizontal_points_coordinates_x,
            "y_points": list_vertical_points_coordinates
        }
        # координаты хаев-лоев в одном списке
        coordinates_dataframe = pd.DataFrame(coordinates_dataframe_dict)

        # создаем список координат точек по вертикали для хаев и лоев отдельно
        list_vertical_high_points_coordinates = data.High.values.tolist()
        list_vertical_low_points_coordinates = data.Low.values.tolist()

        high_low_coordinates_dataframe_dict = {
            "x_points": list_horizontal_points_coordinates,
            "y_high_points": list_vertical_high_points_coordinates,
            "y_low_points": list_vertical_low_points_coordinates
        }

        high_low_coordinates_dataframe = pd.DataFrame(high_low_coordinates_dataframe_dict)

        return coordinates_dataframe, high_low_coordinates_dataframe

    @staticmethod
    def generate_degrees(source_data, degree):
        """
        Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
        Для каждой степени от 1 до degree возводим x в эту степень
        """
        return np.array([
            source_data ** n for n in range(1, degree + 1)
        ]).T

    def train_polynomial(self, data, x, y, visualise_polynomal=False):
        """
        Генерим данные, тренируем модель
        дополнительно рисуем график
        """
        X = self.generate_degrees(data[x], self.DEGREE)

        model = LinearRegression().fit(X, data[y])
        y_pred = model.predict(X)

        if visualise_polynomal:
            error = mean_squared_error(data[y], y_pred)
            print("Степень полинома %d Ошибка %.10f" % (self.DEGREE, error))

            plt.scatter(data[x], data[y], 40, 'g', 'o', alpha=0.8, label='data')
            plt.plot(data[x][len(data.x_points) // 2:], y_pred[len(y_pred) // 2:])
            plt.show()

        return model.coef_, y_pred

    @staticmethod
    def get_borders(high_low_coordinates_dataframe, high_asymptote, low_asymptote, percent_of_min_max_bars):
        """
        Функция возвращает границы ренджа,
        """
        # количество минимальных и максимальных точек при определении медианы
        count_max_min_bars = int(np.around(len(high_low_coordinates_dataframe) * percent_of_min_max_bars))

        sorted_high_asymptote_diffs = np.sort(high_low_coordinates_dataframe.y_high_points.values - high_asymptote)[
                                      ::-1]  # находим разницы от хаев к асимптотам хаев
        sorted_low_asymptote_diffs = np.sort(low_asymptote - high_low_coordinates_dataframe.y_low_points.values)[
                                     ::-1]  # находим разницы от хаев к асимптотам хаев
        # получаем дельты для асимптот, поиском медианы максимальных отклонений от старой асимптоты
        delta_high_asymptote = np.median(sorted_high_asymptote_diffs[:count_max_min_bars])
        delta_low_asymptote = np.median(sorted_low_asymptote_diffs[:count_max_min_bars])

        high_border = high_asymptote + delta_high_asymptote
        low_border = low_asymptote - delta_low_asymptote

        return high_border[::-1], low_border[::-1]

    @staticmethod
    def fullness_of_range(high_low_coordinates_dataframe, high_asymptote, low_asymptote):

        # максимлаьная заполенность ренджа
        max_area_of_range = sum(high_asymptote - low_asymptote)

        # находим кооддинаты хаев и лоев внутри зоны асимптот, если хай или лоу выходят за границы,
        # то их значения приравниваются границам
        high_bars_coordinates = []
        low_bars_coordinates = []
        for high, low, high_max, low_min in zip(high_low_coordinates_dataframe.y_high_points.values,
                                                high_low_coordinates_dataframe.y_low_points.values, high_asymptote,
                                                low_asymptote):
            if high <= high_max:
                high_bars_coordinates.append(high)
            else:
                high_bars_coordinates.append(high_max)
            if low >= low_min:
                low_bars_coordinates.append(low)
            else:
                low_bars_coordinates.append(low_min)
        high_bars_coordinates = np.array(high_bars_coordinates)
        low_bars_coordinates = np.array(low_bars_coordinates)

        area_of_range = sum(high_bars_coordinates - low_bars_coordinates)
        # anti_area_of_range это переменная которая равна площади баров которые выходят за границы ренджа,
        # и эта величина отнимается от общей площади ренджа
        # то есть ужесточает метрику, чем меньше бары покидают границы ренджа - тем лучше
        anti_area_of_range = np.array(
            [i for i in high_low_coordinates_dataframe.y_high_points.values - high_asymptote if i > 0]).sum() + \
                             np.array([i for i in low_asymptote - high_low_coordinates_dataframe.y_low_points.values if
                                       i > 0]).sum()

        fullness_relation = (area_of_range - anti_area_of_range * 0.5) / max_area_of_range

        return fullness_relation

    def get_atr(self, diffs):
        """
        Функция выполняет расчет ATR из условия, что любое значение из выборки не больше ATR более чем в 3 раза и не
        меньше его более чем в 3 раза
        """
        # отбираю count последних периодов
        atr = diffs.mean()

        high_board = atr * self.HIGH_BORDER <= diffs.max()
        low_board = atr * self.LOW_BORDER >= diffs.min()

        if high_board or low_board:
            # Поиск позиций дивиаций
            deviation_true_false = (diffs >= atr * self.HIGH_BORDER) | (diffs <= atr * self.LOW_BORDER)
            # Инициализируем очищенную дату
            cleaned_data = []
            # Оставляем данные со средним внутридневным движением
            for diff, true_or_false in zip(diffs, deviation_true_false):
                if not true_or_false:
                    cleaned_data.append(diff)
            cleaned_data = np.array(cleaned_data)
            atr = self.get_atr(diffs=cleaned_data)
            return atr
        else:
            return atr

    @staticmethod
    def process_data(data, time_only=True, ignore_volume=False):
        """
        Функция выполняет препроцессинг данных
        """

        time_list = []
        date_time_list = []
        open_price_list = []
        high_price_list = []
        low_price_list = []
        close_price_list = []
        volume_list = []

        for i in range(len(data.columns)):

            index_o = data.columns[i].index("O")
            index_h = data.columns[i].index("H")
            index_l = data.columns[i].index("L")
            index_c = data.columns[i].index("C")
            index_v = data.columns[i].index("V")

            # Производим извлечение данных из строк
            open_price = float(data.columns[i][index_o + 3: index_c - 1])
            high_price = float(data.columns[i][index_h + 3: index_l - 1])
            low_price = float(data.columns[i][index_l + 3: index_v - 1])
            close_price = float(data.columns[i][index_c + 3: index_h - 1])

            str_volume = data.columns[i][index_v + 3: -1]

            if ignore_volume:
                open_price_list.append(open_price)
                high_price_list.append(high_price)
                low_price_list.append(low_price)
                close_price_list.append(close_price)
                volume_list.append(0)

                # Обрабатываем время (приводим к нужному формату и временной зоне)
                time = data.columns[i][data.columns[i].index("[", 1) + 12: data.columns[i].index("[", 1) + 20]
                time = datetime.strptime(time, "%H:%M:%S")
                time = time.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0, minute=0)
                time = time.strftime("%H:%M:%S")
                time_list.append(time)

                date_time = data.columns[i][data.columns[i].index("[", 1) + 1: data.columns[i].index("[", 1) + 20]
                date_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
                date_time = date_time.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0,
                                                                                       minute=0)
                date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
                date_time_list.append(date_time)
            else:
                if "E" in str(str_volume):
                    # добавляем данные в список
                    open_price_list.append(open_price)
                    high_price_list.append(high_price)
                    low_price_list.append(low_price)
                    close_price_list.append(close_price)
                    volume_list.append(0)

                    # Обрабатываем время (приводим к нужному формату и временной зоне)
                    time = data.columns[i][data.columns[i].index("[", 1) + 12: data.columns[i].index("[", 1) + 20]
                    time = datetime.strptime(time, "%H:%M:%S")
                    time = time.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0, minute=0)
                    time = time.strftime("%H:%M:%S")
                    time_list.append(time)

                    date_time = data.columns[i][data.columns[i].index("[", 1) + 1: data.columns[i].index("[", 1) + 20]
                    date_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
                    date_time = date_time.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0,
                                                                                           minute=0)
                    date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
                    date_time_list.append(date_time)
                else:
                    volume = float(str_volume)
                    if volume == 0:
                        continue
                    else:
                        # добавляем данные в список
                        open_price_list.append(open_price)
                        high_price_list.append(high_price)
                        low_price_list.append(low_price)
                        close_price_list.append(close_price)
                        volume_list.append(volume)

                        # Обрабатываем время (приводим к нужному формату и временной зоне)
                        time = data.columns[i][data.columns[i].index("[", 1) + 12: data.columns[i].index("[", 1) + 20]
                        time = datetime.strptime(time, "%H:%M:%S")
                        time = time.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0, minute=0)
                        time = time.strftime("%H:%M:%S")
                        time_list.append(time)

                        date_time = data.columns[i][
                                    data.columns[i].index("[", 1) + 1: data.columns[i].index("[", 1) + 20]
                        date_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
                        date_time = date_time.astimezone(pytz.timezone("Europe/Kiev")).replace(microsecond=0, second=0,
                                                                                               minute=0)
                        date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")
                        date_time_list.append(date_time)

        if time_only:
            processed_dict = {
                "Open": open_price_list,
                "High": high_price_list,
                "Low": low_price_list,
                "Close": close_price_list,
                "Volume": volume_list
            }
            processed_dataframe = pd.DataFrame(data=processed_dict, index=time_list)

        else:
            processed_dict = {
                "Open": open_price_list,
                "High": high_price_list,
                "Low": low_price_list,
                "Close": close_price_list,
                "Volume": volume_list
            }
            processed_dataframe = pd.DataFrame(data=processed_dict, index=date_time_list)

        return processed_dataframe


while True:
    # если файл, наличие которого говорит об обновлении исторических данных, то выполняем обработку данных
    if os.path.exists(
            "/content/drive/MyDrive/TC/data_is_downloaded.txt"):
        range_ = range_detection()
        range_()
        range_.plot_all_ranges()
        # Удаляем файл для того чтобы цикл обработки данных не повторялся до получения новых данных
        os.remove("/content/drive/MyDrive/TC/data_is_downloaded.txt")
        time_.sleep(10)
    else:
        # запускаем код каждые 10 секунд
        time_.sleep(10)