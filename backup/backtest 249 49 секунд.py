#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Бэктестинг стратегии статистического арбитража на основе коинтеграции криптовалютных пар.
"""

# Установка переменных окружения для многопоточности OpenBLAS
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Отключаем многопоточность BLAS для избежания конфликтов с Numba
os.environ['OMP_NUM_THREADS'] = '1'        # Отключаем многопоточность OpenMP
os.environ['MKL_NUM_THREADS'] = '1'        # Отключаем многопоточность MKL

# Импорт необходимых библиотек
import logging
import warnings
import importlib.machinery
import importlib.util
import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
from scipy import stats  # Нужен для расчета p-value в тесте на коинтеграцию
import bottleneck as bn  # Для быстрого скользящего среднего и стандартного отклонения
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import test_precise
from test_precise import fast_coint_numba_final, precompute_differences
import ast

# Ограничение количества потоков для Numba
import numba
from numba import njit, prange
numba.set_num_threads(4)  # Установка максимального количества потоков для Numba

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Импорт функции fast_coint_numba_final из test_precise.py
loader = importlib.machinery.SourceFileLoader('test_precise', 'test_precise.py')
spec = importlib.util.spec_from_loader('test_precise', loader)
test_precise = importlib.util.module_from_spec(spec)
loader.exec_module(test_precise)
fast_coint_numba_final = test_precise.fast_coint_numba_final

# Функция для эффективного расчета скользящего среднего и стандартного отклонения с использованием алгоритма Welford
@numba.njit(fastmath=True, cache=True)
def compute_coint_and_beta(log_x, log_y, W, min_count=None):
    """
    JIT-функция для предварительного вычисления массивов таурис (для коинтеграции) и бета.
    
    Args:
        log_x (numpy.ndarray): Логарифмические цены первого актива
        log_y (numpy.ndarray): Логарифмические цены второго актива
        W (int): Размер окна
        min_count (int, optional): Минимальное количество ненулевых наблюдений
        
    Returns:
        tuple: (tau_lengths, beta_lengths) - массивы значений тау и бета коэффициентов
    """
    N = len(log_x)
    if min_count is None:
        min_count = int(W * 0.9)
    
    # Инициализируем массивы для результатов
    tau_lengths = np.full(N, np.nan, dtype=np.float64)
    beta_lengths = np.full(N, np.nan, dtype=np.float64)
    
    # Обрабатываем каждое окно от W до N
    for i in range(W, N):
        # Выделяем окно
        window_start = i - W
        window_end = i
        
        x_window = log_x[window_start:window_end]
        y_window = log_y[window_start:window_end]
        
        # Удаляем NaN
        mask = ~(np.isnan(x_window) | np.isnan(y_window))
        x_window_clean = x_window[mask]
        y_window_clean = y_window[mask]
        
        # Проверяем достаточно ли данных
        if len(x_window_clean) < min_count or len(y_window_clean) < min_count:
            tau_lengths[i] = np.nan
            beta_lengths[i] = np.nan
            continue
        
        # Вычисляем бета-коэффициент: beta = (x @ y) / (y @ y)
        sum_yy = np.dot(y_window_clean, y_window_clean)
        if sum_yy == 0:
            tau_lengths[i] = np.nan
            beta_lengths[i] = np.nan
            continue
            
        sum_xy = np.dot(x_window_clean, y_window_clean)
        beta = sum_xy / sum_yy
        
        # Запоминаем бета
        beta_lengths[i] = beta
        
        # Вычисляем tau (статистику коинтеграции) вне зависимости от p-value
        try:
            # Используем fast_coint_numba_final с предварительно вычисленными суммами
            tau, _, _ = fast_coint_numba_final(x_window_clean, y_window_clean, sum_yy=sum_yy, sum_xy=sum_xy)
            tau_lengths[i] = tau
        except Exception:
            tau_lengths[i] = np.nan
    
    return tau_lengths, beta_lengths


@numba.njit(fastmath=True, cache=True)
def welford_moving_stats(data, window_size, min_count=None):
    """
    Рассчитывает скользящее среднее и стандартное отклонение за один проход используя алгоритм Welford.
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
        window_size (int): Размер скользящего окна
        min_count (int, optional): Минимальное количество ненулевых наблюдений для расчета (если None, то равно window_size)
    
    Returns:
        tuple: (mu_array, sigma_array) - массивы скользящих средних и стандартных отклонений
    """
    n = len(data)
    if min_count is None:
        min_count = window_size
    
    # Создаем выходные массивы
    mu_array = np.full(n, np.nan)
    sigma_array = np.full(n, np.nan)
    
    for i in range(window_size - 1, n):
        # Выделяем окно
        window = data[i - window_size + 1:i + 1]
        
        # Удаляем NaN
        valid_data = window[~np.isnan(window)]
        
        # Проверяем, достаточно ли валидных данных
        if len(valid_data) < min_count:
            continue
        
        # Вычисляем среднее и стандартное отклонение
        mu_array[i] = np.mean(valid_data)
        sigma_array[i] = np.std(valid_data)
    
    return mu_array, sigma_array

# Шаг 3: Чтение данных
def load_data():
    """Загрузка и подготовка данных для бэктестинга."""
    logging.info("Чтение файла historical_data_clean.parquet...")
    prices_raw = pd.read_parquet('historical_data_clean.parquet')
    
    logging.info(f"Прочитано {prices_raw.shape[0]} строк и {prices_raw.shape[1]} столбцов")
    
    # Преобразование timestamp в datetime, если необходимо
    if prices_raw['timestamp'].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(prices_raw['timestamp']):
        logging.info("Преобразование timestamp в datetime...")
        prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"], errors="raise")
    
    # Установка timestamp в индекс, сортировка и приведение частоты
    prices = prices_raw.set_index('timestamp').sort_index()
    prices = prices.pivot(columns='symbol', values='close').asfreq('15T').ffill(limit=1)
    
    logging.info(f"Подготовленный DataFrame содержит {prices.shape[0]} строк и {prices.shape[1]} столбцов")
    return prices


# Шаг 4: Чтение файла с парами
def read_pairs(file_path='Pairs.txt'):
    """Чтение файла с парами и преобразование в список кортежей."""
    logging.info(f"Чтение файла с парами: {file_path}")
    try:
        with open(file_path, 'r') as f:
            pairs = ast.literal_eval(f.read())
        logging.info(f"Прочитано {len(pairs)} пар")
        return pairs
    except Exception as e:
        logging.error(f"Ошибка при чтении файла с парами: {e}")
        return []

# Шаг 5: Преобразование цен в логарифмы
def convert_to_log_prices(prices):
    """Преобразование цен в логарифмы."""
    logging.info("Преобразование цен в логарифмы...")
    # Заменяем нулевые и отрицательные цены на NaN перед преобразованием в логарифмы
    prices_filtered = prices.copy()
    prices_filtered[prices_filtered <= 0] = np.nan
    
    # Преобразование в логарифмы
    log_prices = np.log(prices_filtered)
    
    # Проверка на наличие NaN и бесконечностей
    nan_count = log_prices.isna().sum().sum()
    logging.info(f"Количество NaN в логарифмических ценах: {nan_count}")
    
    return log_prices


def prepare_price_arrays(log_px, pairs):
    """
    Подготовка оптимизированных numpy-массивов для каждой пары символов.
    
    Args:
        log_px (pandas.DataFrame): DataFrame с логарифмическими ценами
        pairs (list): Список пар для бэктестинга в формате кортежей (symbol1, symbol2)
        
    Returns:
        dict: Словарь, где ключи - кортежи (symbol1, symbol2), а значения - numpy-массивы shape=(T, 2)
    """
    logging.info("Подготовка оптимизированных numpy-массивов для пар...")
    
    # Создаем словарь для хранения массивов
    pair_arrays = {}
    
    # Преобразуем DataFrame с логарифмическими ценами в numpy-массив
    # Это позволит избежать затрат pandas на индексацию в дальнейшем
    log_px_np = log_px.values  # Получаем numpy-массив значений
    
    # Создаем словарь для быстрого доступа к индексам колонок
    col_indices = {col: i for i, col in enumerate(log_px.columns)}
    
    # Создаем массивы для каждой пары
    total_pairs = len(pairs)
    for i, (sym1, sym2) in enumerate(pairs):
        if i % 1000 == 0 and i > 0:
            logging.info(f"Подготовлено {i} из {total_pairs} пар")
            
        # Проверяем, что оба символа есть в данных
        if sym1 not in col_indices or sym2 not in col_indices:
            continue
            
        # Получаем индексы колонок
        idx1 = col_indices[sym1]
        idx2 = col_indices[sym2]
        
        # Создаем двумерный массив shape=(T, 2)
        # где первый столбец - логарифмы цен sym1, второй - логарифмы цен sym2
        arr = np.stack((log_px_np[:, idx1], log_px_np[:, idx2]), axis=1)
        
        # Сохраняем массив в словарь
        pair_arrays[(sym1, sym2)] = arr
    
    logging.info(f"Подготовлено {len(pair_arrays)} пар из {total_pairs}")
    return pair_arrays, log_px.index

# Шаг 7: Функция для расчета бета-коэффициента
def calc_beta(x, y, use_kalman=False, delta=1e-4):
    """
    Расчет бета-коэффициента для пары временных рядов.
    
    Args:
        x (numpy.ndarray): Первый временной ряд (зависимая переменная)
        y (numpy.ndarray): Второй временной ряд (независимая переменная)
        use_kalman (bool): Использовать ли Kalman фильтр вместо OLS
        delta (float): Параметр забывания для Kalman фильтра
        
    Returns:
        float: Бета-коэффициент
    """
    # Удаляем NaN из данных
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask].values if hasattr(x, 'values') else x[mask]
    y_clean = y[mask].values if hasattr(y, 'values') else y[mask]
    
    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan
    
    if use_kalman:
        # Реализация Kalman фильтра для оценки бета
        # Инициализация
        n = len(x_clean)
        beta = 0.0  # Начальное значение бета
        P = 1000.0  # Начальная ковариация
        R = 0.001  # Шум измерения
        
        # Цикл Kalman фильтра
        for i in range(n):
            # Предсказание
            x_pred = beta * y_clean[i]
            
            # Обновление ковариации
            P = P + delta
            
            # Вычисление Kalman gain
            K = P * y_clean[i] / (y_clean[i]**2 * P + R)
            
            # Обновление бета
            beta = beta + K * (x_clean[i] - x_pred)
            
            # Обновление ковариации
            P = (1 - K * y_clean[i]) * P
        
        return beta
    else:
        # Простой OLS: beta = (x @ y) / (y @ y)
        return np.dot(x_clean, y_clean) / np.dot(y_clean, y_clean)


# Шаг 8: Основной цикл бэктестинга для одной пары
def backtest_one_pair(pair, price_data, params, dates_index=None, use_kalman=False):
    """
    Бэктестинг стратегии статистического арбитража для одной пары, с использованием оптимизированных numpy-массивов.
    
    Args:
        pair (tuple): Пара символов (symbol1, symbol2)
        price_data: Может быть либо:
                    - dict: Словарь с numpy-массивами для каждой пары (оптимизированный вариант)
                    - pandas.DataFrame: Логарифмические цены (устаревший вариант)
        params (BacktestParams): Параметры бэктестинга
        dates_index (pd.DatetimeIndex, optional): Индекс дат для numpy-массивов
        use_kalman (bool): Использовать ли Kalman фильтр вместо OLS для расчета бета-коэффициента
        
    Returns:
        pandas.DataFrame: Результаты бэктестинга (сделки)
    """
    import time
    start_time = time.time()
    
    symbol1, symbol2 = pair
    logging.info(f"Бэктестинг пары {symbol1}-{symbol2}")
    
    # Проверяем тип входных данных и получаем массивы цен соответствующим образом
    if isinstance(price_data, dict):
        # Используем оптимизированный вариант с предварительно подготовленными numpy-массивами
        if pair not in price_data:
            logging.warning(f"Пара {symbol1}-{symbol2} отсутствует в подготовленных данных")
            return pd.DataFrame()
            
        # Получаем двумерный массив [время, [log_price1, log_price2]]
        pair_array = price_data[pair]
        log_px1_np = pair_array[:, 0]
        log_px2_np = pair_array[:, 1]
        dates = dates_index.values if dates_index is not None else None
    else:
        # Устаревший вариант с DataFrame
        # Проверка, что оба символа есть в данных
        if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
            logging.warning(f"Один или оба символа отсутствуют в данных: {symbol1}, {symbol2}")
            return pd.DataFrame()  # Возвращаем пустой DataFrame
        
        # Извлекаем цены и логарифмы цен для пары в numpy-массивы для ускорения
        log_px1_np = price_data[symbol1].values
        log_px2_np = price_data[symbol2].values
        dates = price_data.index.values
        
    # Предварительно вычисляем массивы tau и beta для всех окон
    rolling_window = params.rolling_window
    min_count = int(rolling_window * 0.9)
    tau_lengths, beta_lengths = compute_coint_and_beta(log_px1_np, log_px2_np, rolling_window, min_count)
    # Преобразуем логарифмические цены в обычные цены
    prices1_np = np.exp(log_px1_np)
    prices2_np = np.exp(log_px2_np)
    
    # Список для хранения сделок
    trades = []
    
    # Флаги для отслеживания состояния позиции
    position_open = False
    position_side = 0  # 1 = long x/short y, -1 = short x/long y
    
    # Переменные для хранения параметров позиций
    entry_index = 0
    entry_time = None
    entry_prices = (0, 0)
    entry_quantities = (0, 0)
    hold_bars = 0
    entry_beta = 0
    entry_spread = 0
    entry_z = 0
    
    # Основной цикл бэктестинга
    for i in range(rolling_window, len(log_px1_np)):
        # Используем предварительно вычисленные значения tau и beta из массивов
        tau = tau_lengths[i]
        beta = beta_lengths[i]
        
        # Пропускаем, если tau или beta имеют значения NaN
        if np.isnan(tau) or np.isnan(beta):
            continue
            
        # Преобразовать tau в p_value привычным способом
        p_value = stats.norm.sf(abs(tau)) * 2  # Двусторонний тест
        
        # Если p_value выше порога, пропустить текущий timestamp
        if p_value > params.p_value_threshold:
            logging.debug(f"Пропуск бара из-за высокого p-value ({p_value:.4f} > {params.p_value_threshold}) для пары {symbol1}-{symbol2}")
            continue
        
        # Текущие значения логарифмов цен и цен
        price1 = prices1_np[i]
        price2 = prices2_np[i]
        
        # Проверка на отрицательные или нулевые цены
        if price1 <= 0 or price2 <= 0:
            logging.debug(f"Пропуск бара из-за некорректных цен ({price1}, {price2}) для пары {symbol1}-{symbol2}")
            continue
        
        # Установка параметров окна
        W = params.rolling_window
        min_count_w = int(W * 0.9)
        
        # Получаем текущее значение спреда (без создания полного spread_array)
        spread = log_px1_np[i] - beta * log_px2_np[i]
        
        # Инициализируем параметры статистики Велфорда при первом использовании бета
        if i == rolling_window or not hasattr(params, 'welford_stats') or params.welford_stats.get(pair) is None:
            # Вычисление начальных сумм для окна W:2*W
            if i+1 >= 2*W:
                initial_window = np.array([log_px1_np[W:2*W] - beta * log_px2_np[W:2*W]])
                valid_values = initial_window[~np.isnan(initial_window)]
                sum_s = np.sum(valid_values)
                sum_ss = np.sum(valid_values**2)
                window_size = len(valid_values)
            else:
                # Если недостаточно данных, используем доступное окно
                available_size = i - W + 1
                if available_size <= 0:
                    # Не хватает данных, пропускаем расчет
                    mu = np.nan
                    sigma = np.nan
                    continue
                
                initial_window = np.array([log_px1_np[W:i+1] - beta * log_px2_np[W:i+1]])
                valid_values = initial_window[~np.isnan(initial_window)]
                sum_s = np.sum(valid_values)
                sum_ss = np.sum(valid_values**2)
                window_size = len(valid_values)
                
            if not hasattr(params, 'welford_stats'):
                params.welford_stats = {}
                
            # Сохраняем статистики Велфорда для текущей пары
            params.welford_stats[pair] = {
                'sum_s': sum_s,
                'sum_ss': sum_ss,
                'window_size': window_size,
                'last_beta': beta,
                'spreads': np.array([log_px1_np[i-W:i] - beta * log_px2_np[i-W:i]])
            }
        
        # Получаем текущие статистики Велфорда для этой пары
        welford = params.welford_stats[pair]
        
        # Проверяем, изменилась ли бета (в этом случае нужно пересчитать окно)
        if welford['last_beta'] != beta:
            # Пересчитываем для нового значения бета
            initial_window = np.array([log_px1_np[i-W:i] - beta * log_px2_np[i-W:i]])
            valid_values = initial_window[~np.isnan(initial_window)]
            welford['sum_s'] = np.sum(valid_values)
            welford['sum_ss'] = np.sum(valid_values**2)
            welford['window_size'] = len(valid_values)
            welford['last_beta'] = beta
            welford['spreads'] = initial_window
            
        # Обновляем окно по алгоритму Велфорда O(1)
        if i >= W:
            # Значение, которое выходит из окна
            s_out = log_px1_np[i-W] - beta * log_px2_np[i-W]
            
            # Проверяем, не было ли NaN для выходящего значения
            if not np.isnan(s_out):
                welford['sum_s'] -= s_out
                welford['sum_ss'] -= s_out**2
                welford['window_size'] -= 1
            
            # Проверяем, не является ли новое значение NaN
            if not np.isnan(spread):
                welford['sum_s'] += spread
                welford['sum_ss'] += spread**2
                welford['window_size'] += 1
        
        # Рассчитываем статистики из обновленных сумм
        if welford['window_size'] < min_count_w:
            mu = np.nan
            sigma = np.nan
        else:
            mu = welford['sum_s'] / welford['window_size']
            var = (welford['sum_ss'] - welford['window_size'] * mu**2) / welford['window_size']
            sigma = np.sqrt(max(var, 0))  # Защита от отрицательной дисперсии из-за численных ошибок
        
        # Проверка на слишком маленькое стандартное отклонение или NaN
        if np.isnan(sigma) or sigma < 1e-6:
            logging.debug(f"Пропуск бара из-за некорректного стандартного отклонения спреда для пары {symbol1}-{symbol2}")
            continue
        
        # Расчет z-score
        z = (spread - mu) / sigma
        
        # Шаг 9: Если |z| > params.z_open и при этом позиции ещё нет, открыть позицию
        # с зафиксированными qty = params.notional_size / price, используя рассчитанный beta и текущее значение spread
        
        # Проверка на наличие открытой позиции
        if not position_open:
            # Проверка на сигнал открытия позиции
            if z > params.z_open:
                # Открытие позиции: шорт x / лонг y
                position_side = -1
                position_open = True
                entry_index = i
                entry_time = dates[i]
                entry_prices = (price1, price2)
                
                # Расчет количества для каждой ноги и фиксация параметров
                qty1 = params.notional_size / price1
                qty2 = params.notional_size / price2
                entry_quantities = (qty1, qty2)
                entry_beta = beta  # Фиксируем бета на момент входа
                entry_spread = spread  # Фиксируем спред на момент входа
                entry_z = z  # Фиксируем z-score на момент входа
                
                logging.debug(f"Открытие позиции для пары {symbol1}-{symbol2}: шорт {symbol1} / лонг {symbol2}, z-score = {z:.2f}")
                
            elif z < -params.z_open:
                # Открытие позиции: лонг x / шорт y
                position_side = 1
                position_open = True
                entry_index = i
                entry_time = dates[i]
                entry_prices = (price1, price2)
                
                # Расчет количества для каждой ноги и фиксация параметров
                qty1 = params.notional_size / price1
                qty2 = params.notional_size / price2
                entry_quantities = (qty1, qty2)
                entry_beta = beta  # Фиксируем бета на момент входа
                entry_spread = spread  # Фиксируем спред на момент входа
                entry_z = z  # Фиксируем z-score на момент входа
                
                logging.debug(f"Открытие позиции для пары {symbol1}-{symbol2}: лонг {symbol1} / шорт {symbol2}, z-score = {z:.2f}")
        
        # Если позиция открыта, проверяем условия закрытия
        if position_open:
            hold_bars += 1
            
            # Проверяем условия закрытия: z-score вернулся к среднему или превышен макс. срок удержания
            if (abs(z) < params.z_close) or (hold_bars > params.max_hold_bars):
                # Закрытие позиции
                exit_time = dates[i]
                exit_prices = (price1, price2)
                
                # Используем зафиксированные при открытии позиции количества активов
                qty1, qty2 = entry_quantities
                
                # Расчет комиссий на основе фактического объема сделки
                # Используем зафиксированные количества активов
                # Комиссия при входе (на основе цен входа)
                commission_entry_x = abs(qty1 * entry_prices[0] * params.fee_taker)
                commission_entry_y = abs(qty2 * entry_prices[1] * params.fee_taker)
                
                # Комиссия при выходе (на основе цен выхода)
                commission_exit_x = abs(qty1 * price1 * params.fee_taker)
                commission_exit_y = abs(qty2 * price2 * params.fee_taker)
                
                # Общая комиссия
                fees = commission_entry_x + commission_entry_y + commission_exit_x + commission_exit_y
                
                # Расчет P&L с учетом стороны позиции
                if position_side == 1:  # Лонг x / шорт y
                    pnl_x = qty1 * (price1 - entry_prices[0])
                    pnl_y = qty2 * (entry_prices[1] - price2)
                else:  # Шорт x / лонг y
                    pnl_x = qty1 * (entry_prices[0] - price1)
                    pnl_y = qty2 * (price2 - entry_prices[1])
                    
                pnl = pnl_x + pnl_y - fees
                
                # Добавляем запись о сделке с фиксированными параметрами
                
                # Рассчитываем длительность сделки в часах (совместимо с numpy.timedelta64)
                duration_timedelta = exit_time - entry_time
                # Преобразуем в наносекунды, затем в секунды, затем в часы
                duration_in_hours = duration_timedelta.astype('timedelta64[ns]').astype(float) / (1e9 * 3600.0)
                
                trade = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'hold_bars': hold_bars,
                    'side': position_side,
                    'entry_price1': entry_prices[0],
                    'entry_price2': entry_prices[1],
                    'exit_price1': exit_prices[0],
                    'exit_price2': exit_prices[1],
                    'qty1': entry_quantities[0],
                    'qty2': entry_quantities[1],
                    'entry_beta': entry_beta,  # Фиксированная бета при входе
                    'entry_spread': entry_spread,  # Фиксированный спред при входе
                    'entry_z': entry_z,  # Фиксированный z-score при входе
                    'exit_spread': spread,  # Спред при выходе
                    'pnl': pnl,
                    'fees': fees,
                    'duration': duration_in_hours  # Длительность сделки в часах
                }
                trades.append(trade)
                
                # Сброс флагов позиции
                position_open = False
                position_side = 0
                hold_bars = 0
                
                # Логируем закрытие позиции
                logging.debug(f"Закрыта позиция по паре {symbol1}-{symbol2} при z = {z:.2f}, P&L = {pnl:.2f}")
        
        # Если позиция не открыта, проверяем условия открытия
        elif abs(z) > params.z_open:
            # Проверка цен на NaN при открытии позиции
            if np.isnan(price1) or np.isnan(price2) or price1 <= 0 or price2 <= 0:
                logging.debug(f"Невозможно открыть позицию из-за недопустимых цен пары {symbol1}-{symbol2} на {current_time}")
                continue
                
            # Шаг 15: Проверка на дублирование сделок
            # Если z при повторном i даёт сигнал уже открытой позиции, не пытаемся заново открывать
            # Это дополнительная проверка, т.к. флаг position_open должен уже предотвращать это
            if position_open:
                logging.debug(f"Позиция уже открыта для пары {symbol1}-{symbol2}, пропускаем сигнал на открытие")
                continue
                
            # Открытие позиции
            position_open = True
            entry_index = i
            entry_time = current_time
            entry_prices = (price1, price2)
            
            # Фиксируем beta при открытии позиции
            entry_beta = beta
            entry_spread = spread
            entry_z = z
            
            # Определяем направление позиции
            if z < 0:  # Спред ниже среднего - лонг x, шорт y
                position_side = 1
            else:  # Спред выше среднего - шорт x, лонг y
                position_side = -1
            
            # Расчет количества активов для каждой ноги
            # Используем фиксированный notional_size для каждой ноги
            qty1 = params.notional_size / price1
            qty2 = params.notional_size / price2
            entry_quantities = (qty1, qty2)
            
            logging.debug(f"Открыта позиция по паре {symbol1}-{symbol2} при z = {z:.2f}, сторона: {position_side}")
    
    # Рассчитываем и выводим время выполнения бэктеста
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Создаем DataFrame с результатами
    if not trades:
        logging.info(f"Нет сделок для пары {symbol1}-{symbol2}, P&L: 0.00, время: {execution_time:.2f} сек.")
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades)
    
    # Добавляем информацию о паре
    trades_df['pair'] = f"{symbol1}-{symbol2}"
    
    total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0.0
    logging.info(f"Всего сделок для пары {symbol1}-{symbol2}: {len(trades_df)}, P&L: {total_pnl:.2f}, время: {execution_time:.2f} сек.")
    
    # Добавляем время выполнения в DataFrame
    trades_df['execution_time'] = execution_time
    
    return trades_df

# Шаг 6: Параметры симуляции
class BacktestParams:
    """Класс для хранения параметров бэктестинга."""
    def __init__(self):
        # Размер скользящего окна (60 дней по 4 бара в час)
        self.rolling_window = 60 * 24 * 4  # 5760 точек (60 дней × 24 часа × 4 бара в час)

        # Пороги для z-score
        self.z_open = 2.0   # Порог для открытия позиции
        self.z_close = 0.5  # Порог для закрытия позиции

        # Порог для p-value в тесте на коинтеграцию
        self.p_value_threshold = 0.05  # Позиция открывается только если p-value <= 0.05

        # Максимальное время удержания позиции (в барах)
        self.max_hold_bars = 20 * 24 * 4  # 20 дней по 15-минутных баров

        # Комиссии Bybit для фьючерсов
        self.fee_maker = 0.0002  # 0.02% для мейкера
        self.fee_taker = 0.00055  # 0.055% для тейкера

        # Размер позиции (в USDT)
        self.notional_size = 1000.0  # Фиксированный размер позиции в USDT на каждую ногу
        
        # Начальный капитал
        self.initial_capital = 10000.0  # Начальный капитал в USDT
        
        # Период для бэктестинга
        self.start_date = None  # Если None, используются все доступные данные с начала
        self.end_date = None    # Если None, используются все доступные данные до конца
        
        # Флаг для использования Kalman фильтра вместо OLS
        self.use_kalman = False
        
        # Параметры Kalman фильтра (если используется)
        self.delta = 1e-4  # Скорость забывания в Kalman фильтре
        
        # Число потоков для параллельных вычислений
        self.n_jobs = -1  # -1 означает использование всех доступных ядер
        
        # Порог p-value для теста на коинтеграцию
        self.p_value_threshold = 0.05  # Значение по умолчанию

# Шаг 10: Добавление функции для расчета метрик и визуализации
def calculate_performance_metrics(trades_df, initial_capital=10000.0):
    """
    Расчет метрик производительности стратегии.
    
    Args:
        trades_df (pandas.DataFrame): Датафрейм с результатами бэктестинга
        initial_capital (float): Начальный капитал
        
    Returns:
        tuple: (метрики, кривая капитала, просадка в процентах)
    """
    if trades_df.empty:
        empty_series = pd.Series(dtype='float64')
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_trade_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_abs': 0,
            'calmar_ratio': 0,
            'avg_duration': 0
        }, empty_series, empty_series
    
    # Убедимся, что 'exit_time' отсортирован перед расчетом cumsum
    trades_df = trades_df.sort_values(by='exit_time')
    
    # Создаем кривую капитала
    equity_curve = pd.Series(initial_capital + trades_df['pnl'].cumsum().values, index=trades_df['exit_time'])
    
    # Расчет максимальной просадки
    running_max = equity_curve.cummax()
    drawdown_values = equity_curve - running_max  # Просадка в денежных единицах
    
    # Инициализируем drawdown_percent с нулями той же длины, что и equity_curve
    drawdown_percent = pd.Series(0.0, index=equity_curve.index, dtype='float64')
    
    # Рассчитываем drawdown_percent только для тех случаев, где running_max не равен нулю
    # и не равен initial_capital (если initial_capital был 0, это предотвратит деление на 0)
    # или если running_max просто не 0.
    # Более безопасный способ - проверить, что running_max не равен 0.
    non_zero_running_max_mask = running_max != 0
    drawdown_percent[non_zero_running_max_mask] = (drawdown_values[non_zero_running_max_mask] / running_max[non_zero_running_max_mask]) * 100
    
    max_drawdown_abs = abs(drawdown_values.min()) if not drawdown_values.empty else 0
    max_drawdown_pct = abs(drawdown_percent.min()) if not drawdown_percent.empty else 0

    # Расчет дневной доходности для Sharpe ratio
    daily_returns = equity_curve.resample('D').last().ffill().pct_change().dropna()
    
    # Расчет Sharpe ratio (годовой)
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 and daily_returns.std() != 0 else 0
    
    # Расчет Calmar ratio (годовая доходность / макс. просадка в процентах)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days if len(equity_curve) > 1 else 0
    calmar_ratio = 0
    if days > 0 and max_drawdown_pct > 0:
        # Используем initial_capital для расчета общей доходности, чтобы избежать проблем если equity_curve.iloc[0] равен 0
        if initial_capital != 0:
            total_return_percent = ((equity_curve.iloc[-1] / initial_capital) - 1) * 100
            annual_return_percent = total_return_percent * (365.0 / days)
            calmar_ratio = annual_return_percent / max_drawdown_pct
        elif equity_curve.iloc[0] != 0 : # Fallback если initial_capital = 0 но первая точка эквити не 0
            total_return_percent = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
            annual_return_percent = total_return_percent * (365.0 / days)
            calmar_ratio = annual_return_percent / max_drawdown_pct
            
    # Сбор всех метрик
    metrics = {
        'total_trades': len(trades_df),
        'total_pnl': trades_df['pnl'].sum(),
        'win_rate': 100 * (trades_df['pnl'] > 0).mean() if len(trades_df) > 0 else 0,
        'avg_trade_pnl': trades_df['pnl'].mean() if len(trades_df) > 0 else 0,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_abs': max_drawdown_abs,
        'calmar_ratio': calmar_ratio,
        'avg_duration': trades_df['duration'].mean() if 'duration' in trades_df and len(trades_df) > 0 else 0
    }
    
    return metrics, equity_curve, drawdown_percent

def backtest_all_pairs(pairs, price_data, params, n_jobs=1, dates_index=None):
    """
    Бэктестинг для всех пар с использованием оптимизированных numpy-массивов.
    
    Args:
        pairs (list): Список пар в формате (symbol1, symbol2)
        price_data: Может быть либо:
                    - dict: Словарь с numpy-массивами для каждой пары (оптимизированный вариант)
                    - pandas.DataFrame: Логарифмические цены (устаревший вариант)
        params (BacktestParams): Параметры бэктестинга
        n_jobs (int): Количество параллельных процессов
        dates_index (pd.DatetimeIndex, optional): Индекс дат для numpy-массивов
        
    Returns:
        pandas.DataFrame: Все сделки для всех пар
    """
    import time
    total_start_time = time.time()
    
    logging.info(f"Запуск бэктестинга для {len(pairs)} пар с использованием {n_jobs} процессов")
    
    # Используем joblib с потоковым бэкендом для параллельного выполнения
    # Так как код внутри пары использует Numba/NumPy (GIL освобождён), потоки эффективнее процессов
    # Это экономит RAM (нет копирования price_data) и время сериализации
    results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(delayed(backtest_one_pair)(pair, price_data, params, dates_index) for pair in pairs)
    
    # Фильтруем непустые результаты
    non_empty_results = [df for df in results if not df.empty]
    
    if not non_empty_results:
        logging.warning("Нет результатов бэктестинга для всех пар")
        return pd.DataFrame()  # Возвращаем пустой DataFrame
    
    # Объединяем все результаты в один DataFrame
    all_trades = pd.concat(non_empty_results, ignore_index=True)
    
    # Рассчитываем общее время выполнения
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # Для статистики не требуется таблица - вся информация уже выведена в логах для каждой пары
    
    logging.info(f"Всего сделок: {len(all_trades)}")
    logging.info(f"Общее время выполнения: {total_execution_time:.2f} сек.")
    
    return all_trades

if __name__ == "__main__":
    logging.info("Начало выполнения скрипта...")
    
    # Шаг 3: Загрузка данных
    prices = load_data()
    
    # Вывод информации о загруженных данных
    logging.info(f"Диапазон дат: {prices.index.min()} - {prices.index.max()}")
    logging.info(f"Количество символов: {prices.columns.size}")
    logging.debug(f"Первые 5 строк:\n{prices.head()}")
    
    # Шаг 4: Чтение пар
    pairs = read_pairs()
    if pairs:
        logging.info(f"Прочитано {len(pairs)} пар")
        logging.debug(f"Первые 5 пар: {pairs[:5]}")
    
    # Шаг 5: Преобразование в логарифмы
    log_px = convert_to_log_prices(prices)
    logging.debug(f"Первые 5 строк логарифмических цен:\n{log_px.head()}")
    
    # Шаг 6: Создание параметров бэктестинга
    params = BacktestParams()
    logging.info(f"Параметры бэктестинга:")
    logging.info(f"Размер окна: {params.rolling_window} точек")
    logging.info(f"Пороги z-score: открытие {params.z_open}, закрытие {params.z_close}")
    logging.info(f"Комиссии: maker {params.fee_maker*100}%, taker {params.fee_taker*100}%")
    logging.info(f"Размер позиции: {params.notional_size} USDT на ногу")
    
    # Шаг 9: Добавление функции бэктестинга всех пар
    
    # Подготовка оптимизированных numpy-массивов для всех пар
    pair_arrays, dates_index = prepare_price_arrays(log_px, pairs)
    
    # Тестируем функцию расчета бета-коэффициента:
    test_pair = ("ETHUSDT", "XYMUSDT")
    test_window_start = 0
    test_window_end = params.rolling_window
    
    # Проверка, что пара есть в словаре
    if test_pair in pair_arrays:
        test_array = pair_arrays[test_pair]
        x_test = test_array[test_window_start:test_window_end, 0]
        y_test = test_array[test_window_start:test_window_end, 1]
    else:
        # Используем старый способ, если пары нет в словаре
        x_test = log_px[test_pair[0]].iloc[test_window_start:test_window_end].values
        y_test = log_px[test_pair[1]].iloc[test_window_start:test_window_end].values
    
    # Очистка от NaN
    mask = ~(np.isnan(x_test) | np.isnan(y_test))
    x_test_clean = x_test[mask]
    y_test_clean = y_test[mask]
    
    # Тестируем бэктестинг на одной паре
    logging.info("Тестирование бэктестинга на одной паре:")
    trades_df = backtest_one_pair(test_pair, pair_arrays, params, dates_index=dates_index)
    if not trades_df.empty:
        logging.info(f"Получено {len(trades_df)} сделок для пары {test_pair[0]}-{test_pair[1]}")
        logging.debug(f"Первые сделки:\n{trades_df[['entry_time', 'exit_time', 'side', 'pnl', 'entry_z', 'entry_beta', 'entry_spread']].head()}")
    
    # Запуск бэктестинга на первых 10 парах с использованием параллельной обработки
    # Задаем фиксированное значение 2 процесса, чтобы избежать конфликта с Numba
    n_jobs_safe = 2  # Фиксированное значение для предотвращения конфликта с Numba
    logging.info(f"Запуск бэктестинга на первых 10 парах с использованием {n_jobs_safe} процессов...")
    all_trades = backtest_all_pairs(pairs[:10], pair_arrays, params, n_jobs=n_jobs_safe, dates_index=dates_index)
    
    if not all_trades.empty:
        # Расчет метрик производительности
        metrics, equity_curve, drawdown = calculate_performance_metrics(all_trades, params.initial_capital)
        
        # Вывод метрик
        logging.info("Метрики производительности:")
        logging.info(f"Всего сделок: {metrics['total_trades']}")
        logging.info(f"Общий P&L: {metrics['total_pnl']:.2f} USDT")
        logging.info(f"Средний P&L на сделку: {metrics['avg_trade_pnl']:.2f} USDT")
        logging.info(f"Винрейт: {metrics['win_rate']:.2f}%")
        logging.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"Максимальная просадка: {metrics['max_drawdown_pct']:.2f}%")
        logging.info(f"Calmar ratio: {metrics['calmar_ratio']:.2f}")
        logging.info(f"Средняя длительность сделки: {metrics['avg_duration']:.2f} часов")
        
        # Визуализация результатов
        # Используем функцию из модуля visualization.py
        try:
            from visualization import plot_results, save_results_to_csv, plot_equity_curve
            plot_results(all_trades, equity_curve, drawdown)
            plot_equity_curve(equity_curve, 'equity_curve.png')
            save_results_to_csv(all_trades, metrics)
            logging.info("Визуализация и сохранение результатов завершены")
        except ImportError:
            logging.warning("Модуль visualization.py не найден. Визуализация не будет выполнена.")
        except Exception as e:
            logging.error(f"Ошибка при визуализации: {e}")
        
        # Сохранение результатов в CSV
        all_trades.to_csv('backtest_trades.csv', index=False)
        logging.info("Результаты сохранены в backtest_trades.csv")
    
    logging.info("Завершение выполнения скрипта.")