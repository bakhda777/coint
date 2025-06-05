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

# Стандартные библиотеки
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from itertools import combinations
import json
import logging
import math
import multiprocessing
from pathlib import Path
from typing import Optional
import sys
import time
import warnings

# Сторонние библиотеки
import bottleneck as bn
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numba

# Глобальные переменные для риск-менеджмента в бэктестинге
GLOBAL_OPEN_TRADES_COUNT = 0  # Счетчик открытых сделок
GLOBAL_EQUITY = 10000.0       # Начальный капитал (будет обновляться при каждой сделке)
GLOBAL_EQUITY_CURVE = []      # История изменения капитала


def reset_risk_management_globals(initial_equity: float = 10000.0) -> None:
    """
    Сбрасывает глобальные переменные риск-менеджмента на начальные значения.
    
    Args:
        initial_equity (float): Начальный капитал для бэктеста.
    """
    global GLOBAL_OPEN_TRADES_COUNT, GLOBAL_EQUITY, GLOBAL_EQUITY_CURVE
    GLOBAL_OPEN_TRADES_COUNT = 0
    GLOBAL_EQUITY = initial_equity
    GLOBAL_EQUITY_CURVE = []
from numba import njit, prange
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import minimize

# Локальные импорты
from core import cointegration
from core import data
from core import prices
from core import reports
from utils_path import list_partition_symbols

# Ограничение количества потоков для Numba
numba.set_num_threads(4)  # Установка максимального количества потоков для Numba

# Используем DataProvider и другие функции из core.data

# CLI Helper Functions
def parse_date_optional(date_str: str | None) -> datetime | None:
    """Парсит строку даты в формате YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS в объект datetime.
    Возвращает None, если строка пустая или None."""
    if not date_str:
        return None
    try:
        if 'T' in date_str:
            # Попытка парсинга с учетом временной зоны, если она есть
            dt_obj = datetime.fromisoformat(date_str)
            # Если нет информации о временной зоне, считаем UTC по умолчанию или локальной в зависимости от требований
            # Для CLI обычно лучше работать с aware datetimes или четко документировать предположения
            # Здесь, если fromisoformat успешно спарсил, но объект naive, можно его локализовать или оставить как есть
            return dt_obj
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        # Логируем ошибку для отладки, но пользователю выдаем ArgumentTypeError
        logging.debug(f"Ошибка парсинга даты '{date_str}': {e}")
        raise argparse.ArgumentTypeError(f"Неверный формат даты: '{date_str}'. Ожидается YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS. Пример: 2023-01-01 или 2023-01-01T14:30:00")

def read_symbols_from_file(filepath: str | Path) -> list[str] | None:
    """Читает список символов из текстового файла (один символ на строку)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        if not symbols:
            logging.warning(f"Файл символов {filepath} пуст или содержит только комментарии.")
            return []
        return symbols
    except FileNotFoundError:
        logging.warning(f"Файл символов не найден: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Ошибка при чтении файла символов {filepath}: {e}")
        return None

def write_symbols_to_file(symbols: list[str], filepath: str | Path) -> None:
    """Записывает список символов в текстовый файл (один символ на строку)."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        logging.info(f"Список из {len(symbols)} символов сохранен в {filepath}")
    except Exception as e:
        logging.error(f"Ошибка при записи файла символов {filepath}: {e}")

def read_pairs_from_file(filepath: str | Path) -> list[tuple[str, str]] | None:
    """Читает список пар из JSON-файла. Ожидается список списков/кортежей по 2 элемента."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs_data = json.load(f)
        if not isinstance(pairs_data, list):
            logging.error(f"Ошибка формата в файле пар {filepath}: ожидается JSON массив (список). Получено: {type(pairs_data)}")
            return None
        
        pairs = []
        for item in pairs_data:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], str):
                pairs.append(tuple(item))
            else:
                logging.warning(f"Пропущен некорректный элемент в файле пар {filepath}: {item}. Ожидался список/кортеж из двух строк.")
        
        if not pairs and pairs_data: # Если исходный список был не пуст, но валидных пар не нашлось
            logging.warning(f"В файле пар {filepath} не найдено корректных пар.")
            return [] # Возвращаем пустой список, если файл был, но не содержал валидных пар
        elif not pairs_data: # Если исходный файл был пуст (валидный JSON '[]')
             logging.info(f"Файл пар {filepath} пуст.")
             return []
        return pairs
    except FileNotFoundError:
        logging.warning(f"Файл пар не найден: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка декодирования JSON в файле пар {filepath}: {e}")
        return None
    except Exception as e:
        logging.error(f"Неожиданная ошибка при чтении файла пар {filepath}: {e}")
        return None

def write_pairs_to_file(pairs: list[tuple[str, str]], filepath: str | Path) -> None:
    """Записывает список пар в JSON-файл."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Преобразуем кортежи в списки для более стандартного JSON вывода
        list_of_lists = [list(pair) for pair in pairs]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(list_of_lists, f, indent=4)
        logging.info(f"Список из {len(pairs)} пар сохранен в {filepath}")
    except Exception as e:
        logging.error(f"Ошибка при записи файла пар {filepath}: {e}")



# Шаг 6: Параметры симуляции (перемещено выше для разрешения NameError)
class BacktestParams:
    """Класс для хранения параметров бэктестинга."""
    def __init__(self):
        # Размер скользящего окна (60 дней по 4 бара в час)
        self.rolling_window = 60 * 24 * 4  # 5760 точек (60 дней × 24 часа × 4 бара в час)

        # Пороги для z-score
        self.z_open = 2.0   # Порог для открытия позиции
        self.z_close = 0.5  # Порог для закрытия позиции

        # Пороги для tau-статистики теста на коинтеграцию остатков (ADF-тест на стационарность спреда)
        self.tau_open_threshold = -3.0    # Порог tau-статистики для открытия позиции (более отрицательное -> сильнее коинтеграция)
        self.tau_close_threshold = -2.0   # Порог tau-статистики для закрытия позиции (если коинтеграция ослабла)

        # Фильтры для входа в позицию
        self.use_tau_filter_for_entry = True       # Использовать ли фильтр по tau-статистике для входа
        self.use_zscore_for_entry = False          # Использовать ли z-score вместо tau для входа/выхода
        self.use_zscore_for_exit = False           # Использовать ли z-score для выхода из позиции
        # Параметры p-value фильтра (могут использоваться совместно с tau-фильтром)
        self.use_pvalue_filter_for_entry = False   # Использовать ли фильтр по p-value для входа
        self.p_value_threshold_entry = 0.05      # Порог p-value для входа (если use_pvalue_filter_for_entry=True)

        # Параметры отбора пар
        self.max_corr_neighbors = 15        # Максимальное количество коррелированных соседей на символ для предварительного отбора
        self.use_correlation_filter = True  # Флаг для использования корреляционного фильтра
        self.tau_prefilter = -2.8           # Порог tau-статистики для предварительного фильтра пар
        self.top_pairs = 100                # Количество лучших пар для отбора по силе коинтеграции (tau-статистике)
        self.max_universe_size_by_metric = 500 # Максимальный размер вселенной по метрике (ликвидность/капитализация)

        # Максимальный лаг для ADF-теста в compute_coint_and_beta и calculate_coint_params
        self.adf_max_lag = 12 # Этот параметр передается в compute_coint_and_beta как adf_maxlag

        # Новые параметры, связанные с количеством данных и ADF
        self.min_data_points_for_coint: int = 30  # Мин. точек для расчета коинтеграции (EG-тест)
        self.min_data_points_for_adf_test: int = 20  # Мин. точек для ADF-теста на остатках/спреде
        self.min_data_points_for_hurst: int = 50    # Мин. точек для расчета Hurst exponent
        self.rolling_adf_window_size: int = 30    # Размер скользящего окна для ADF в backtest_one_pair (можно связать с min_data_points_for_coint)
        


        # Максимальное время удержания позиции (в барах)
        self.max_hold_bars = 30 * 24 * 4  # 30 дней по 15-минутных баров

        # Защитный стоп-лосс (в USDT)
        self.max_loss_per_trade = 1000.0  # Максимальный убыток на одну сделку

        # Комиссии для фьючерсов
        self.maker_fee = 0.0002  # 0.02% для мейкера (maker)
        self.taker_fee = 0.00055  # 0.055% для тейкера (taker)
        
        # Для обратной совместимости
        self.fee_maker = self.maker_fee
        self.fee_taker = self.taker_fee
        
        # Фандинг (funding rate) для бессрочных контрактов
        self.funding_apr_estimate = 0.05  # 5% годовых
        
        # Коэффициент для расчета проскальзывания: slip = k · (notional / ADV)
        self.slip_k = 0.1  # Коэффициент проскальзывания
        
        # Флаг для включения/выключения учета проскальзывания
        self.use_slippage = True  # Если True, рассчитывается проскальзывание
        
        # Средний дневной объем торгов (Average Daily Volume) для расчета проскальзывания
        self.default_adv = 1_000_000.0  # Значение по умолчанию, если не предоставлено реальное значение
        
        # Флаг для включения/выключения учета фандинга
        self.use_funding = True  # Если True, рассчитывается фандинг

        # Размер позиции (в USDT)
        self.notional_size = 1000.0  # Базовый размер позиции в USDT на каждую ногу

        # Флаг для позиционирования с учетом волатильности
        self.volatility_sizing = True  # Если True, размер позиции обратно пропорционален волатильности

        # Минимальный возраст токена в днях для его использования в стратегии
        self.min_token_age_days = 180  # Минимальный возраст токена в днях (6 месяцев)

        # Контроль портфельного риска
        self.max_concurrent_trades = 20  # Максимальное количество одновременных открытых позиций
        self.max_capital_usage = 0.8  # Максимальный процент использования капитала (80%)

        # Начальный капитал
        self.initial_capital = 10000.0  # Начальный капитал в USDT

        # Имена колонок в данных (для гибкости, если они отличаются от стандартных)
        self.price_column_name = "close"    # Колонка с ценами закрытия
        # self.volume_column_name = "volume"   # Удалено дублирующееся определение, см. ниже volume_column_name для ADV
        self.adv_column_name = "turnover" # Колонка со среднесуточным оборотом (в USDT, если используется для фильтрации)

        # Параметры параллелизма
        # self.n_jobs = 4                     # Удалено дублирующееся определение, см. ниже n_jobs = -1

        # Период для бэктестинга
        self.start_date = None  # Если None, используются все доступные данные с начала
        self.end_date = None    # Если None, используются все доступные данные до конца

        # Минимальное стандартное отклонение спреда
        self.min_spread_std = 1e-5  # Минимальное значение std спреда для открытия позиции

        # Флаг для использования Kalman фильтра вместо OLS
        self.use_kalman = False

        # Параметры Kalman фильтра (если используется)
        self.delta = 1e-4  # Скорость забывания в Kalman фильтре

        # Число потоков для параллельных вычислений
        self.n_jobs = -1  # -1 означает использование всех доступных ядер

        # Минимальный среднесуточный объем торгов в USDT для фильтрации ликвидности
        self.min_adv_usdt: float | None = None

        # Имена колонок для расчета ADV (Average Daily Volume)
        self.volume_column_name: str = "volume_usdt"  # Имя колонки с объемом в USDT
        self.price_column_name_for_adv: str = "close"    # Имя колонки с ценой закрытия для расчета ADV

        # Пути для сохранения графиков
        self.equity_curve_output_path: Path | None = None
        self.trades_dist_output_path: Path | None = None

        # Максимальный допустимый гэп для заполнения цен
        self.max_gap_minutes = 60  # Максимальный гэп в минутах для forward fill

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ожидаемые колонки для DataFrame с результатами сделок
EXPECTED_RESULT_COLUMNS = [
    'timestamp', 'pair', 'position_type', 'entry_price_1', 'entry_price_2',
    'exit_price_1', 'exit_price_2', 'qty_1', 'qty_2', 'pnl', 'pnl_percent',
    'commission', 'spread_at_entry', 'spread_at_exit', 'beta_at_entry',
    'tau_stat_at_entry', 'z_score_at_entry', 'holding_period_bars',
    'exit_reason', 'initial_capital_at_trade', 'equity_after_trade',
    'execution_time' # Добавлено в backtest_one_pair
]


# Функция fetch_series перенесена в модуль core.prices


# Функция get_log_prices перенесена в модуль core.prices



# Функция filter_symbols_by_liquidity была удалена, так как её функциональность
# должна быть реализована в модуле core.data

# Функция _available_symbols перенесена в core.data


# Функция calculate_pair_beta_and_tau_static перенесена в модуль core.cointegration



@lru_cache(maxsize=128)
def compute_coint_and_beta(
    pair: tuple[str, str],
    start_dt: datetime,
    end_dt: datetime,
    params: BacktestParams,
    check_hurst: bool = False
) -> dict[str, Any] | None:
    """
    Рассчитывает различные метрики коинтеграции для пары символов.
    
    Эта функция является оберткой для cointegration.compute_coint_and_beta, которая работает
    с данными цен с помощью prices.get_log_prices и передает их в core.cointegration.

    Args:
        pair (tuple[str, str]): Пара символов (symbol1, symbol2).
        start_dt (datetime): Начальная дата периода.
        end_dt (datetime): Конечная дата периода.
        params (BacktestParams): Параметры бэктеста.
        check_hurst (bool): Флаг, указывающий, нужно ли рассчитывать экспоненту Хёрста.

    Returns:
        dict[str, Any] | None: Словарь с результатами или None в случае ошибки.
        Ключи словаря: 'pair', 'beta', 'tau_stat', 'is_coint', 'n_obs',
                       'half_life', 'z_score', 'hurst' (если check_hurst=True),
                       'adf_pvalue', 'optimal_lag', 'coint_error_msg',
                       'mu_spread', 'sigma_spread' (статистики спреда для z-score).
    """
    # Инициализация результата с информацией о паре
    result: dict[str, Any] = {
        'pair': pair,
        'beta': None,
        'tau_stat': None,
        'is_coint': False,
        'n_obs': 0,
        'half_life': None,
        'z_score': None,
        'hurst': None,
        'adf_pvalue': None,
        'optimal_lag': None,
        'coint_error_msg': None,
        'mu_spread': None,
        'sigma_spread': None,
    }

    log_prices_data = prices.get_log_prices(pair, start_dt, end_dt, params.max_gap_minutes)

    if log_prices_data is None:
        result['coint_error_msg'] = f"No log prices for {pair} in {start_dt}-{end_dt}"
        # logging.debug(result['coint_error_msg'])
        return result # Возвращаем словарь с ошибкой, а не None, для консистентности

    log_prices_array, _ = log_prices_data
    result['n_obs'] = log_prices_array.shape[1]

    if result['n_obs'] < params.min_data_points_for_coint:
        result['coint_error_msg'] = f"Insufficient data points ({result['n_obs']}) for {pair} in {start_dt}-{end_dt}. Need {params.min_data_points_for_coint}."
        # logging.debug(result['coint_error_msg'])
        return result

    try:
        # Получаем временные ряды
        y_series = log_prices_array[0, :]  # Первый инструмент - зависимая переменная
        x_series = log_prices_array[1, :]  # Второй инструмент - независимая переменная

        # Вызываем функцию из модуля core.cointegration
        coint_result = cointegration.compute_coint_and_beta(
            y_series=y_series,
            x_series=x_series,
            params=params,
            regression_coint_trend='c'  # 'c' для ADF на остатках с константой
        )

        # Копируем основные результаты из core.cointegration.compute_coint_and_beta
        result['beta'] = coint_result['beta']
        result['tau_stat'] = coint_result['tau_stat']
        result['is_coint'] = coint_result['is_coint']
        result['adf_pvalue'] = coint_result['adf_pvalue']
        result['optimal_lag'] = coint_result['optimal_lag']
        result['coint_error_msg'] = coint_result['error_msg']
        
        # Расчет статистик спреда для z-score (всегда, независимо от коинтеграции)
        spread_values = y_series - result['beta'] * x_series
        if len(spread_values) > 0:
            result['mu_spread'] = float(np.mean(spread_values))
            result['sigma_spread'] = float(np.std(spread_values))
            # Защита от деления на ноль
            if result['sigma_spread'] == 0 or np.isnan(result['sigma_spread']):
                result['sigma_spread'] = 1e-8  # Очень маленькое значение вместо 0
        
        # Если есть ошибка, или не коинтегрированы, возвращаем результат сейчас
        if result['coint_error_msg'] or not result['is_coint']:
            return result

        # Расчет экспоненты Хёрста, если требуется
        if check_hurst:
            try:
                from hurst import compute_Hc
                spread = y_series - result['beta'] * x_series
                if len(spread) >= params.min_data_points_for_hurst:  # должно быть не меньше min_data_points_for_hurst точек
                    H, _, _ = compute_Hc(spread, kind='price', simplified=True)
                    result['hurst'] = float(H) if not np.isnan(H) else None
            except Exception as e:
                # logging.warning(f"Failed to compute Hurst exponent for {pair}: {e}")
                result['hurst'] = None
                result['coint_error_msg'] = f"Error calculating Hurst for {pair}: {e}"
        
        return result

    except LinAlgError as lae:
        result['coint_error_msg'] = f"Linear algebra error (e.g., singular matrix) for {pair} in {start_dt}-{end_dt}: {lae}"
        # logging.warning(result['coint_error_msg'])
        # Обнуляем ключевые метрики, если они могли быть установлены до ошибки
        result['beta'], result['tau_stat'], result['is_coint'], result['half_life'], result['z_score'] = None, None, False, None, None 
    except ValueError as ve:
        result['coint_error_msg'] = f"ValueError (e.g., NaNs/Infs in data, insufficient data for statsmodels) for {pair} in {start_dt}-{end_dt}: {ve}"
        # logging.warning(result['coint_error_msg'])
        result['beta'], result['tau_stat'], result['is_coint'], result['half_life'], result['z_score'] = None, None, False, None, None 
    except Exception as e:
        result['coint_error_msg'] = f"Generic error in coint/beta calc for {pair} in {start_dt}-{end_dt}: {e}"
        # logging.error(result['coint_error_msg'], exc_info=True) # Можно добавить exc_info для детального трейсбека
        result['beta'], result['tau_stat'], result['is_coint'], result['half_life'], result['z_score'] = None, None, False, None, None 

    return result


def backtest_one_pair(
    pair: tuple[str, str],
    formation_start_dt: datetime,
    formation_end_dt: datetime,
    trading_start_dt: datetime,
    trading_end_dt: datetime,
    params: BacktestParams,
    update_globals: bool = True  # Флаг для обновления глобальных параметров риск-менеджмента
) -> dict[str, Any]:
    """
    Выполняет бэктест для одной пары символов.

    1. Рассчитывает параметры коинтеграции (бета, тау, z-score, half-life) на формационном периоде.
    2. Если пара коинтегрирована (tau_stat < params.tau_prefilter), проходит по торговому периоду.
    3. В торговом периоде использует БЕТУ, полученную на формационном периоде (статическая бета).
    4. Z-score пересчитывается на каждом шаге на основе текущего спреда и статической беты.
       (Примечание: в оригинальной логике z-score для входа/выхода мог быть основан на статистиках формационного периода,
        но для динамического бэктеста обычно z-score спреда пересчитывается).
       Для простоты восстановления, предположим, что z-score спреда (текущее значение - среднее)/std 
       рассчитывается на основе статистики спреда за формационный период.
    5. Генерирует сигналы на вход/выход на основе z-score и порогов params.z_score_open/close.
       (Или, если используется tau для торговли, то на основе params.tau_open_threshold/tau_close_threshold).
       Судя по предыдущим наработкам, используются tau-пороги для открытия/закрытия.
    6. Учитывает максимальное время удержания позиции (max_holding_period_bars).
    7. Собирает статистику по сделкам.

    Args:
        pair (tuple[str, str]): Пара символов.
        formation_start_dt (datetime): Начало формационного периода.
        formation_end_dt (datetime): Конец формационного периода.
        trading_start_dt (datetime): Начало торгового периода.
        trading_end_dt (datetime): Конец торгового периода.
        params (BacktestParams): Параметры бэктеста.

    Returns:
        dict[str, Any]: Словарь с результатами бэктеста для пары.
                        Включает 'pair', 'trades', 'pnl', 'formation_metrics', 'error_message'.
    """
    # Объявление глобальных переменных в начале функции
    global GLOBAL_OPEN_TRADES_COUNT, GLOBAL_EQUITY, GLOBAL_EQUITY_CURVE
    
    symbol1, symbol2 = pair
    # Локальная копия текущего equity для этого бэктеста
    current_equity = GLOBAL_EQUITY if update_globals else params.initial_capital
    
    # Результаты бэктеста для этой пары
    results = {
        'pair': pair,
        'trades': [], # Список сделок
        'num_trades': 0,
        'final_pnl_pct': 0.0,
        'formation_metrics': None,
        'error_message': None,
        'risk_management': {
            'current_equity': current_equity,
            'open_trades_count': GLOBAL_OPEN_TRADES_COUNT if update_globals else 0,
            'max_concurrent_trades': params.max_concurrent_trades,
            'max_capital_usage': params.max_capital_usage,
            'max_loss_per_trade': params.max_loss_per_trade,
            'volatility_sizing': params.volatility_sizing
        }
    }

    # 1. Расчеты на формационном периоде
    # logging.debug(f"[{pair}] Формационный период: {formation_start_dt} - {formation_end_dt}")
    formation_metrics = compute_coint_and_beta(
        pair, formation_start_dt, formation_end_dt, params, check_hurst=False
    )
    results['formation_metrics'] = formation_metrics

    if formation_metrics is None or formation_metrics.get('beta') is None or formation_metrics.get('tau_stat') is None:
        results['error_message'] = f"[{pair}] Не удалось рассчитать параметры на формационном периоде. Metrics: {formation_metrics}"
        # logging.warning(results['error_message'])
        return results

    beta_formation = formation_metrics['beta']
    tau_stat_formation = formation_metrics['tau_stat']
    # half_life_formation = formation_metrics.get('half_life') # Может быть None
    # z_score_formation_last = formation_metrics.get('z_score') # z-score на конец формационного периода
    
    # Получаем статистики спреда с формационного периода для z-score
    mu_spread_formation = formation_metrics.get('mu_spread', 0.0)
    sigma_spread_formation = formation_metrics.get('sigma_spread', 1.0)
    
    # Дополнительная защита от неправильных значений
    if sigma_spread_formation == 0 or np.isnan(sigma_spread_formation) or sigma_spread_formation < params.min_spread_std:
        sigma_spread_formation = max(params.min_spread_std, 1e-8)

    # Проверка на предварительный фильтр по tau-статистике
    if tau_stat_formation >= params.tau_prefilter: # tau_prefilter обычно отрицательный, чем меньше, тем лучше
        results['error_message'] = f"[{pair}] Не прошла предварительный фильтр: tau_stat формации ({tau_stat_formation:.4f}) >= tau_prefilter ({params.tau_prefilter:.4f})"
        # logging.debug(results['error_message'])
        return results
    
    # logging.info(f"[{pair}] Прошла формационный тест. Beta: {beta_formation:.4f}, Tau: {tau_stat_formation:.4f}, HL: {formation_metrics.get('half_life')}")

    # 2. Подготовка данных для торгового периода
    # logging.debug(f"[{pair}] Торговый период: {trading_start_dt} - {trading_end_dt}")
    trading_log_prices_data = prices.get_log_prices(pair, trading_start_dt, trading_end_dt, params.max_gap_minutes)

    if trading_log_prices_data is None:
        results['error_message'] = f"[{pair}] Нет данных для торгового периода {trading_start_dt} - {trading_end_dt}."
        # logging.warning(results['error_message'])
        return results

    trading_log_prices_array, trading_index = trading_log_prices_data
    log_prices_s1_trading = trading_log_prices_array[0, :]
    log_prices_s2_trading = trading_log_prices_array[1, :]

    if len(log_prices_s1_trading) == 0 or len(log_prices_s2_trading) == 0:
        results['error_message'] = f"[{pair}] Пустые массивы цен в торговом периоде."
        # logging.warning(results['error_message'])
        return results

    # 3. Инициализация переменных для цикла бэктеста
    in_position: bool = False
    position_type: str | None = None  # 'long_spread' (s1 long, s2 short) or 'short_spread' (s1 short, s2 long)
    entry_price_s1: float = 0.0
    entry_price_s2: float = 0.0
    # entry_spread_value: float = 0.0 # Значение спреда при входе
    entry_ts: pd.Timestamp | None = None
    bars_in_trade: int = 0
    # current_pnl_pct: float = 0.0 # PnL текущей сделки, не агрегированный
    
    # Переменные для риск-менеджмента
    daily_pnls = []  # Для расчета Sharpe ratio и MDD
    
    # Локальная копия текущего equity для этого бэктеста
    current_equity = GLOBAL_EQUITY if update_globals else 10000.0  # Если не обновляем глобальные переменные, используем начальное значение
    
    # Для расчета margin используемого в этой паре
    current_pair_margin = 0.0  # Текущая сумма маржи, занятой позицией для этой пары
    stop_loss_threshold = 0.0  # Будет рассчитываться при входе в позицию
    entry_z_score = None  # Z-score при входе в позицию

    # Статистики спреда из формационного периода уже получены в mu_spread_formation и sigma_spread_formation

    # 4. Цикл по торговому периоду
    # logging.debug(f"[{pair}] Начинаем торговый цикл. Всего баров: {len(trading_index)}")
    for i in range(len(trading_index)):
        current_ts = trading_index[i]
        current_s1_log_price = log_prices_s1_trading[i]
        current_s2_log_price = log_prices_s2_trading[i]

        if pd.isna(current_s1_log_price) or pd.isna(current_s2_log_price):
            # logging.debug(f"[{pair}] Пропуск бара {current_ts}: NaN в ценах ({current_s1_log_price}, {current_s2_log_price})")
            if in_position: # Если мы в позиции, а данные стали NaN, это проблема
                bars_in_trade += 1 # Считаем бар, но PnL не изменится
            continue

        # Рассчитываем текущее значение спреда, используя бету с формационного периода
        current_spread_value = current_s1_log_price - beta_formation * current_s2_log_price

        if not in_position:
            # Логика входа в позицию
            # Нужны данные для расчета ADF на скользящем окне, заканчивающемся текущим баром (i)
            # Окно должно быть достаточной длины, например, params.min_data_points_for_coint
            # или специальный параметр params.rolling_adf_window_size
            rolling_window_size = params.rolling_adf_window_size # Используем этот параметр для окна ADF
            
            if i >= rolling_window_size -1: # Убедимся, что у нас достаточно данных для окна
                # Данные для окна: от (i - rolling_window_size + 1) до i включительно
                window_log_s1 = log_prices_s1_trading[i - rolling_window_size + 1 : i + 1]
                window_log_s2 = log_prices_s2_trading[i - rolling_window_size + 1 : i + 1]
                
                # Рассчитываем остатки спреда на этом окне с использованием beta_formation
                window_residuals = window_log_s1 - beta_formation * window_log_s2

                if len(window_residuals) < params.min_data_points_for_adf_test: # params.min_data_points_for_adf_test
                    # logging.debug(f"[{pair}] {current_ts}: Недостаточно данных в скользящем окне ({len(window_residuals)}) для ADF теста. Требуется {params.min_data_points_for_adf_test}")
                    continue

                try:
                    # Вычисляем tau-статистику (ADF-статистику) и p-value для остатков
                    # fast_adf_test_numba(series: np.ndarray, max_lag: int, regression: str = 'c') 
                    # Возвращает (tau_stat, p_value, n_lags, critical_values_approx, ic, ols_res)
                    current_adf_stat, current_pvalue, _, _, _, _ = cointegration.fast_adf_test_numba(window_residuals, params.adf_max_lag, regression='c')
                except Exception as e_adf:
                    # logging.warning(f"[{pair}] {current_ts}: Ошибка при расчете ADF для входа: {e_adf}")
                    continue

                # Проверка условий входа
                
                # Проверка по p-value, если включена
                if params.use_pvalue_filter_for_entry and current_pvalue > params.p_value_threshold_entry:
                    # p-value слишком высокое - коинтеграция недостаточно статистически значима
                    continue
                    
                # Проверки риск-менеджмента перед открытием позиции
                risk_check_passed = True
                risk_rejection_reason = ""
                
                # 1. Проверка на максимальное количество открытых сделок
                if update_globals and GLOBAL_OPEN_TRADES_COUNT >= params.max_concurrent_trades:
                    risk_check_passed = False
                    risk_rejection_reason = f"Достигнут лимит одновременных сделок ({GLOBAL_OPEN_TRADES_COUNT}/{params.max_concurrent_trades})"
                
                # 2. Расчет потенциальной маржи для этой позиции
                s1_price = np.exp(current_s1_log_price)
                s2_price = np.exp(current_s2_log_price)
                
                # Базовый размер позиции
                notional_s1 = params.notional_size
                notional_s2 = beta_formation * params.notional_size
                
                # Корректируем размеры позиций с учетом волатильности, если включено
                if params.volatility_sizing and i >= rolling_window_size:
                    # Рассчитываем волатильность на основе скользящего окна
                    vol_window_size = min(30, rolling_window_size)  # Размер окна для расчета волатильности
                    
                    # Логарифмические доходности
                    s1_returns = np.diff(log_prices_s1_trading[i-vol_window_size:i+1])
                    s2_returns = np.diff(log_prices_s2_trading[i-vol_window_size:i+1])
                    
                    # Стандартные отклонения доходностей
                    vol_s1 = np.std(s1_returns)
                    vol_s2 = np.std(s2_returns)
                    
                    # Минимальная волатильность для предотвращения деления на очень маленькие числа
                    min_vol = 0.0001
                    vol_s1 = max(vol_s1, min_vol)
                    vol_s2 = max(vol_s2, min_vol)
                    
                    # Рассчитываем коэффициенты инверсии волатильности
                    inv_vol_s1 = 1.0 / vol_s1
                    inv_vol_s2 = 1.0 / vol_s2
                    
                    # Нормализуем коэффициенты, чтобы их сумма была равна 2 
                    # (для сохранения общего размера позиции)
                    sum_inv_vol = inv_vol_s1 + inv_vol_s2
                    weight_s1 = 2.0 * inv_vol_s1 / sum_inv_vol
                    weight_s2 = 2.0 * inv_vol_s2 / sum_inv_vol
                    
                    # Применяем веса к размерам позиций
                    notional_s1 = weight_s1 * params.notional_size
                    notional_s2 = weight_s2 * beta_formation * params.notional_size
                
                # Расчет потенциальной маржи для этой позиции
                potential_margin = notional_s1 + notional_s2
                
                # Суммарная маржа по всем открытым позициям + эта новая позиция
                if update_globals:
                    # Здесь мы бы могли отслеживать глобальную маржу, но ее нет в глобальных переменных
                    # Вместо этого просто сравним potential_margin с допустимым процентом от equity
                    max_allowed_margin = current_equity * params.max_capital_usage
                    if potential_margin > max_allowed_margin:
                        risk_check_passed = False
                        risk_rejection_reason = f"Достигнут лимит использования капитала ({potential_margin:.2f} > {max_allowed_margin:.2f})"
                
                # Расчет Z-score текущего спреда (если включена z-score логика)
                current_z_score = (current_spread_value - mu_spread_formation) / sigma_spread_formation
                
                # Логика входа в позицию
                entry_signal = None
                entry_params = {}
                
                if params.use_zscore_for_entry:
                    # Z-score логика для входа в позицию
                    if current_z_score < -params.z_open and risk_check_passed:
                        # Спред сильно ниже среднего -> покупаем спред (long s1, short s2)
                        entry_signal = 'long_spread'
                        entry_params = {'z_score': current_z_score, 'threshold': -params.z_open}
                    elif current_z_score > params.z_open and risk_check_passed:
                        # Спред сильно выше среднего -> продаем спред (short s1, long s2)
                        entry_signal = 'short_spread'
                        entry_params = {'z_score': current_z_score, 'threshold': params.z_open}
                elif params.use_tau_filter_for_entry:
                                         # Tau-статистика логика для входа в позицию
                     if current_adf_stat < params.tau_open_threshold and risk_check_passed:
                         # Если tau сильно отрицательный, ожидаем возврата спреда вверх (покупаем спред: long s1, short s2)
                         entry_signal = 'long_spread'
                         entry_params = {'tau_stat': current_adf_stat, 'threshold': params.tau_open_threshold}
                     elif current_adf_stat > -params.tau_open_threshold and risk_check_passed:
                         # Если tau сильно положительный, ожидаем возврата спреда вниз (продаем спред: short s1, long s2)
                         entry_signal = 'short_spread'
                         entry_params = {'tau_stat': current_adf_stat, 'threshold': -params.tau_open_threshold}
                
                # Обработка сигнала входа в позицию
                if entry_signal == 'long_spread':
                    in_position = True
                    position_type = 'long_spread' # Покупаем спред (s1 long, s2 short), если он слишком низко
                    entry_price_s1 = s1_price
                    entry_price_s2 = s2_price
                    entry_ts = current_ts
                    bars_in_trade = 1
                    
                    # Сохраняем размеры позиций для каждого актива (с учетом волатильности, если включено)
                    entry_notional_s1 = notional_s1
                    entry_notional_s2 = notional_s2
                    
                    # Увеличиваем счетчик открытых сделок и запоминаем маржу
                    if update_globals:
                        GLOBAL_OPEN_TRADES_COUNT += 1
                    current_pair_margin = potential_margin
                    
                    # Устанавливаем стоп-лосс по размеру капитала
                    stop_loss_threshold = params.max_loss_per_trade * current_equity
                    # Сохраняем z-score для трекинга
                    entry_z_score = current_z_score
                    # logging.info(f"[{pair}] {current_ts}: ВХОД LONG SPREAD. S1={entry_price_s1:.4f}, S2={entry_price_s2:.4f}. Entry signal: {entry_params}")
                
                elif entry_signal == 'short_spread':
                    # Расчет проскальзывания при входе (если включено)
                    entry_slippage_pct_s1 = 0.0
                    entry_slippage_pct_s2 = 0.0
                    
                    if params.use_slippage:
                        # Расчет проскальзывания по линейной модели: slip = k · (notional / ADV)
                        adv_s1 = params.default_adv  # Здесь можно было бы использовать реальные значения ADV для каждого токена
                        adv_s2 = params.default_adv
                        
                        # Расчет проскальзывания в % для каждой ноги
                        entry_slippage_pct_s1 = params.slip_k * (notional_s1 / adv_s1)  # проскальзывание в % для S1
                        entry_slippage_pct_s2 = params.slip_k * (notional_s2 / adv_s2)  # проскальзывание в % для S2
                    
                    # Применяем проскальзывание к ценам входа (лонг покупает дороже, шорт продает дешевле)
                    # Для short_spread: продаем S1 дешевле, покупаем S2 дороже
                    entry_price_s1_with_slip = s1_price * (1 - entry_slippage_pct_s1)  # шортим дешевле
                    entry_price_s2_with_slip = s2_price * (1 + entry_slippage_pct_s2)
                    
                    # Успешный вход, сохраняем данные позиции
                    in_position = True
                    position_type = 'short_spread'  # продаем S1, покупаем S2
                    entry_price_s1 = entry_price_s1_with_slip
                    entry_price_s2 = entry_price_s2_with_slip
                    entry_slippage = {'s1_pct': entry_slippage_pct_s1, 's2_pct': entry_slippage_pct_s2}  # Сохраняем информацию о проскальзывании
                    entry_ts = current_ts
                    bars_in_trade = 1
                    
                    # Сохраняем размеры позиций для каждого актива (с учетом волатильности, если включено)
                    entry_notional_s1 = notional_s1
                    entry_notional_s2 = notional_s2
                    
                    # Увеличиваем счетчик открытых сделок и запоминаем маржу
                    if update_globals:
                        GLOBAL_OPEN_TRADES_COUNT += 1
                    current_pair_margin = potential_margin
                    
                    # Устанавливаем стоп-лосс по размеру капитала
                    stop_loss_threshold = params.max_loss_per_trade * current_equity
                    # Сохраняем z-score для трекинга
                    entry_z_score = current_z_score
                    # logging.info(f"[{pair}] {current_ts}: ВХОД SHORT SPREAD. S1={entry_price_s1:.4f}, S2={entry_price_s2:.4f}. Entry signal: {entry_params}")
                    
                elif not risk_check_passed:
                    # Здесь можно записать в лог причину отклонения трейда
                    # logging.debug(f"[{pair}] {current_ts}: Отклонение трейда: {risk_rejection_reason}")
                    pass
                

        else: # if in_position:
            bars_in_trade += 1
            # Логика выхода из позиции
            exit_reason = None
            
            # Расчет текущего z-score для проверки условий выхода
            current_z_score = (current_spread_value - mu_spread_formation) / sigma_spread_formation
            
            # 1. Проверка на закрытие по z-score (если включена z-score логика для выхода)
            if params.use_zscore_for_exit:
                if position_type == 'long_spread' and current_z_score > -params.z_close:
                    # Для long_spread: выход, если z-score вернулся к среднему (> -z_close)
                    exit_reason = f'zscore_exit_long: {current_z_score:.4f} > {-params.z_close:.4f}'
                elif position_type == 'short_spread' and current_z_score < params.z_close:
                    # Для short_spread: выход, если z-score вернулся к среднему (< +z_close)
                    exit_reason = f'zscore_exit_short: {current_z_score:.4f} < {params.z_close:.4f}'
            
                        # 2. Проверка на закрытие по tau-статистике (если включена tau логика для выхода)
            if exit_reason is None and not params.use_zscore_for_exit:
                # Расчет tau-статистики на скользящем окне для проверки условия выхода
                rolling_window_size = params.rolling_adf_window_size
                if i >= rolling_window_size:
                    window_log_s1 = log_prices_s1_trading[i - rolling_window_size + 1 : i + 1]
                    window_log_s2 = log_prices_s2_trading[i - rolling_window_size + 1 : i + 1]
                    
                    # Рассчитываем остатки спреда на этом окне с использованием beta_formation
                    window_residuals = window_log_s1 - beta_formation * window_log_s2
                    
                    if len(window_residuals) >= params.min_data_points_for_adf_test:
                        try:
                            # Вычисляем tau-статистику (ADF-статистику) для проверки условия выхода
                            exit_adf_stat, exit_pvalue, _, _, _, _ = cointegration.fast_adf_test_numba(
                                window_residuals, params.adf_max_lag, regression='c')
                            
                            # Проверка условий выхода по tau-статистике
                            if position_type == 'long_spread' and exit_adf_stat > params.tau_close_threshold:
                                # Для long_spread: выход, если tau стал слишком высоким (коинтеграция ослабла)
                                exit_reason = f'tau_exit_long: {exit_adf_stat:.4f} > {params.tau_close_threshold:.4f}'
                            elif position_type == 'short_spread' and exit_adf_stat < -params.tau_close_threshold:
                                # Для short_spread: выход, если tau стал слишком низким (коинтеграция ослабла)
                                # Используем модуль для сравнения, так как tau_close_threshold отрицательный
                                exit_reason = f'tau_exit_short: {exit_adf_stat:.4f} < {-params.tau_close_threshold:.4f}'
                        except Exception as e_adf_exit:
                            # Ошибка при расчете ADF для выхода - продолжаем без выхода
                            pass
            
            # 3. Проверка на максимальное время удержания позиции
            if exit_reason is None and bars_in_trade >= params.max_hold_bars:
                exit_reason = f'max_bars: {bars_in_trade}'
            
            # 4. Проверка на стоп-лосс
            # Для этого получаем текущие цены
            current_s1_price = np.exp(log_prices_s1_trading[i])
            current_s2_price = np.exp(log_prices_s2_trading[i])
            
            # Оценка проскальзывания при стоп-лоссе (если включено)
            current_s1_exit_price = current_s1_price
            current_s2_exit_price = current_s2_price
            
            if params.use_slippage:
                # Расчет проскальзывания по линейной модели: slip = k · (notional / ADV)
                slip_s1 = params.slip_k * (entry_notional_s1 / params.default_adv)
                slip_s2 = params.slip_k * (entry_notional_s2 / params.default_adv)
                
                # Применяем проскальзывание при выходе
                if position_type == 'long_spread':  # long s1, short s2
                    current_s1_exit_price = current_s1_price * (1 - slip_s1)  # продаем дешевле
                    current_s2_exit_price = current_s2_price * (1 + slip_s2)  # покупаем дороже
                else:  # short_spread: short s1, long s2
                    current_s1_exit_price = current_s1_price * (1 + slip_s1)  # покупаем дороже
                    current_s2_exit_price = current_s2_price * (1 - slip_s2)  # продаем дешевле
            
            # Расчет теоретического PnL если бы мы закрыли позицию сейчас с учетом проскальзывания
            if position_type == 'long_spread':  # long s1, short s2
                curr_pnl_s1 = (current_s1_exit_price - entry_price_s1) * entry_notional_s1 / entry_price_s1
                curr_pnl_s2 = (entry_price_s2 - current_s2_exit_price) * entry_notional_s2 / entry_price_s2
            else:  # short_spread: short s1, long s2
                curr_pnl_s1 = (entry_price_s1 - current_s1_exit_price) * entry_notional_s1 / entry_price_s1
                curr_pnl_s2 = (current_s2_exit_price - entry_price_s2) * entry_notional_s2 / entry_price_s2
            
            # Общий текущий PnL в абсолютном выражении (USD)
            current_trade_pnl = curr_pnl_s1 + curr_pnl_s2
            
            # Расчет комиссий (вход и выход для обеих ног)
            commission_costs = params.fee_taker * 2 * (entry_notional_s1 + entry_notional_s2)
            
            # Расчет фандинга (если включен)
            funding_costs = 0.0
            if params.use_funding and entry_ts is not None and current_ts is not None:
                # Рассчитываем количество часов в позиции
                holding_hours = (current_ts - entry_ts).total_seconds() / 3600
                
                # Годовая ставка фандинга переведенная в часовую
                hourly_funding_rate = params.funding_apr_estimate / (365 * 24)
                
                # Расчет фандинга для каждой ноги
                # Фандинг для лонгов обычно положительный (платит лонг), для шортов отрицательный (получает шорт)
                # В среднем фандинг считается нулевым, но для более реалистичного бэктеста используется ненулевое значение
                if position_type == 'long_spread':  # long s1, short s2
                    funding_s1 = entry_notional_s1 * hourly_funding_rate * holding_hours  # платеж за лонг S1 (отрицательный)
                    funding_s2 = -entry_notional_s2 * hourly_funding_rate * holding_hours  # получение за шорт S2 (положительный)
                else:  # short_spread: short s1, long s2
                    funding_s1 = -entry_notional_s1 * hourly_funding_rate * holding_hours  # получение за шорт S1 (положительный)
                    funding_s2 = entry_notional_s2 * hourly_funding_rate * holding_hours  # платеж за лонг S2 (отрицательный)
                
                # Суммарные затраты на фандинг
                funding_costs = funding_s1 + funding_s2
            
            # Учитываем комиссии и фандинг
            current_trade_pnl -= commission_costs + funding_costs
            
            # Переводим в проценты от вложенного капитала (total_notional)
            total_notional = entry_notional_s1 + entry_notional_s2  # Суммарный размер позиций с учетом волатильности
            trade_pnl_pct = current_trade_pnl / total_notional if total_notional > 0 else 0
            
            # Проверка стоп-лосса (если включен)
            if exit_reason is None and current_trade_pnl < -stop_loss_threshold and stop_loss_threshold > 0:
                exit_reason = f'stop_loss: ${-current_trade_pnl:.2f} > ${stop_loss_threshold:.2f}'
            
            # 5. Проверка на последний бар торгового периода для принудительного закрытия
            if exit_reason is None and i == len(log_prices_s1_trading) - 1:
                exit_reason = 'end_of_period'
            
            # Если есть причина для выхода, закрываем позицию
            close_position = False
            if exit_reason is not None:
                close_position = True
            
            if close_position:
                # Обновляем глобальные переменные риск-менеджмента
                if update_globals:
                    # Снижаем счетчик открытых сделок
                    GLOBAL_OPEN_TRADES_COUNT = max(0, GLOBAL_OPEN_TRADES_COUNT - 1)  # Не допускаем отрицательных значений
                    
                    # Обновляем глобальный эквити
                    GLOBAL_EQUITY += current_trade_pnl
                
                # Добавляем новую точку в equity curve
                GLOBAL_EQUITY_CURVE.append({
                    'timestamp': current_ts,
                    'equity': GLOBAL_EQUITY,
                    'trade_pnl': current_trade_pnl,
                    'pair': f"{pair[0]}/{pair[1]}",
                    'position_type': position_type,
                    'exit_reason': exit_reason
                })
                
                # Сохраняем информацию о проскальзывании при входе и выходе
                entry_slip = {'s1_pct': 0.0, 's2_pct': 0.0}
                if 'entry_slippage' in locals():
                    entry_slip = entry_slippage
                    
                exit_slip = {'s1_pct': 0.0, 's2_pct': 0.0}
                if params.use_slippage:
                    exit_slip = {
                        's1_pct': params.slip_k * (entry_notional_s1 / params.default_adv),
                        's2_pct': params.slip_k * (entry_notional_s2 / params.default_adv)
                    }
                
                # Запись о сделке с учетом проскальзывания и фандинга
                results['trades'].append({
                    'position': position_type,
                    'entry_date': entry_ts,
                    'entry_price_s1': entry_price_s1,
                    'entry_price_s2': entry_price_s2,
                    'entry_notional_s1': entry_notional_s1,
                    'entry_notional_s2': entry_notional_s2,
                    'exit_date': current_ts,
                    'exit_price_s1': current_s1_exit_price,
                    'exit_price_s2': current_s2_exit_price,
                    'pnl': current_trade_pnl,  # В абсолютном выражении (USDT)
                    'pnl_percent': trade_pnl_pct,  # В процентах от инвестированного капитала
                    'commission': commission_costs,  # Комиссионные расходы
                    'funding': funding_costs,  # Затраты на фандинг
                    'slippage': {
                        'entry': entry_slip,
                        'exit': exit_slip,
                        'total_cost': (entry_slip.get('s1_pct', 0.0) + entry_slip.get('s2_pct', 0.0) + 
                                       exit_slip.get('s1_pct', 0.0) + exit_slip.get('s2_pct', 0.0)) * 
                                      (entry_notional_s1 + entry_notional_s2) / 4  # Приближенная стоимость проскальзывания
                    },
                    'bars_in_trade': bars_in_trade,
                    'holding_hours': (current_ts - entry_ts).total_seconds() / 3600 if entry_ts else 0,  # Длительность в часах
                    'exit_reason': exit_reason,
                    'equity_at_entry': current_equity - current_trade_pnl,  # Значение equity на момент входа
                    'equity_at_exit': current_equity,  # Значение equity на момент выхода
                    'stop_loss_threshold': stop_loss_threshold,  # Уровень стоп-лосса для этой сделки
                    'z_score_at_entry': entry_z_score,  # Z-score при входе
                    'z_score_at_exit': current_z_score,  # Z-score при выходе
                    'spread_stats': {  # Статистики спреда из формационного периода
                        'mu_spread': mu_spread_formation,
                        'sigma_spread': sigma_spread_formation
                    }
                })
                
                results['num_trades'] += 1
                # results['final_pnl_pct'] += trade_pnl_pct # Это неверно для портфеля, PnL должен быть на капитал

                # logging.info(f"[{pair}] {current_ts}: ВЫХОД {position_type}. S1={current_s1_exit_price:.4f}, S2={current_s2_exit_price:.4f}. PnL={trade_pnl_pct*100:.2f}%. Причина: {exit_reason}. Баров: {bars_in_trade}")

                # Сброс состояния позиции
                in_position = False
                position_type = None
                bars_in_trade = 0
                entry_ts = None
    
    # Конец цикла по торговому периоду

    # 5. Расчет итоговых метрик
    if results['trades']:
        # Простой суммарный PnL (не учитывает реинвестирование)
        results['final_pnl_pct'] = sum(trade['pnl_percent'] for trade in results['trades'])
        results['final_pnl_absolute'] = sum(trade['pnl'] for trade in results['trades'])
        
        # Добавляем информацию о риск-менеджменте
        results['risk_management'] = {
            'initial_equity': current_equity - results['final_pnl_absolute'],
            'final_equity': current_equity if update_globals else current_equity + results['final_pnl_absolute'],
            'max_concurrent_trades': params.max_concurrent_trades,
            'max_capital_usage': params.max_capital_usage,
            'max_loss_per_trade': params.max_loss_per_trade,
            'volatility_sizing': params.volatility_sizing
        }
        
        # Если использовались глобальные переменные, добавляем кривую доходности
        if update_globals and GLOBAL_EQUITY_CURVE:
            results['equity_curve'] = GLOBAL_EQUITY_CURVE
        
        # Расчет статистик доходности
        if daily_pnls:
            daily_returns = pd.Series(daily_pnls)
            
            # Статистики доходности
            results['stats'] = {
                'total_trades': len(results['trades']),
                'win_trades': sum(1 for t in results['trades'] if t['pnl'] > 0),
                'loss_trades': sum(1 for t in results['trades'] if t['pnl'] <= 0),
                'avg_trade_pnl': np.mean([t['pnl'] for t in results['trades']]) if results['trades'] else 0,
                'avg_win_pnl': np.mean([t['pnl'] for t in results['trades'] if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in results['trades']) else 0,
                'avg_loss_pnl': np.mean([t['pnl'] for t in results['trades'] if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in results['trades']) else 0,
                'max_win_pnl': max([t['pnl'] for t in results['trades']], default=0),
                'max_loss_pnl': min([t['pnl'] for t in results['trades']], default=0),
                'avg_bars_in_trade': np.mean([t['bars_in_trade'] for t in results['trades']]) if results['trades'] else 0
            }
            
            # Добавляем Sharpe ratio, если есть достаточно данных
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                # Для дневных данных используем корень из 252 (рабочих дней)
                annualization_factor = np.sqrt(252 if params.notional_size > 10 else 365)
                results['stats']['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * annualization_factor

    # logging.info(f"[{pair}] Бэктест завершен. Сделок: {results['num_trades']}, Итоговый PnL: {results['final_pnl_pct']*100:.2f}%")
    return results


def backtest_all_pairs(
    pairs: list[tuple[str, str]], 
    price_data: dict, 
    params: BacktestParams, 
    n_jobs: int = 1,
    window_start_dt: datetime | None = None,
    window_end_dt: datetime | None = None,
    dates_index: list[datetime] | None = None,
    initial_equity: float = 10000.0
) -> dict:
    """
    Запускает бэктест для списка пар с учетом риск-менеджмента.
    
    Args:
        pairs: Список пар для тестирования [(s1, s2), ...]
        price_data: Словарь с ценами для всех символов
        params: Параметры бэктеста
        n_jobs: Количество параллельных процессов (1 = последовательное выполнение)
        window_start_dt: Начальная дата окна бэктеста
        window_end_dt: Конечная дата окна бэктеста
        dates_index: Индекс дат для формирования результатов
        initial_equity: Начальный капитал для риск-менеджмента
        
    Returns:
        Словарь с результатами бэктеста для всех пар
    """
    import time
    import copy
    
    # Сбрасываем глобальные переменные риск-менеджмента перед началом бэктеста
    reset_risk_management_globals(initial_equity=initial_equity)
    
    results = {
        'pair_results': {},
        'total_pnl': 0.0,
        'num_trades': 0,
        'trades': [],
        'execution_time': 0.0,
        'risk_management': {
            'initial_equity': initial_equity,
            'final_equity': initial_equity,  # Будет обновлено в конце
            'equity_curve': [],
            'max_concurrent_trades': params.max_concurrent_trades,
            'max_capital_usage': params.max_capital_usage,
            'max_loss_per_trade': params.max_loss_per_trade,
            'volatility_sizing': params.volatility_sizing
        }
    }
    
    start_time = time.time()
    
    if n_jobs > 1 and len(pairs) > 1:
        # Параллельное выполнение через joblib
        # Важно: при параллельном выполнении глобальные переменные
        # риск-менеджмента не будут корректно обновляться между процессами
        from joblib import Parallel, delayed
        
        # Отключаем обновление глобальных переменных для параллельного режима
        temp_params = copy.deepcopy(params)
        
        pair_results = Parallel(n_jobs=n_jobs)(delayed(backtest_one_pair)(
            pair=pair, 
            price_data=price_data, 
            params=temp_params,
            window_start_dt=window_start_dt,
            window_end_dt=window_end_dt,
            update_globals=False  # Важно: отключаем обновление глобальных переменных
        ) for pair in pairs)
        
        # Соединяем результаты
        for pair, pair_result in zip(pairs, pair_results):
            results['pair_results'][f"{pair[0]}/{pair[1]}"] = pair_result
            if 'trades' in pair_result and pair_result['trades']:
                results['trades'].extend(pair_result['trades'])
                results['num_trades'] += pair_result['num_trades']
                results['total_pnl'] += pair_result['final_pnl_absolute']
    else:
        # Последовательное выполнение с полным учетом риск-менеджмента
        for pair in pairs:
            pair_result = backtest_one_pair(
                pair=pair, 
                price_data=price_data, 
                params=params,
                window_start_dt=window_start_dt,
                window_end_dt=window_end_dt,
                update_globals=True  # Включаем обновление глобальных переменных
            )
            results['pair_results'][f"{pair[0]}/{pair[1]}"] = pair_result
            if 'trades' in pair_result and pair_result['trades']:
                results['trades'].extend(pair_result['trades'])
                results['num_trades'] += pair_result['num_trades']
                results['total_pnl'] += pair_result['final_pnl_absolute']
    
    # Обновляем финальное значение equity
    results['risk_management']['final_equity'] = GLOBAL_EQUITY
    results['risk_management']['equity_curve'] = GLOBAL_EQUITY_CURVE
    
    # Добавляем время выполнения
    results['execution_time'] = time.time() - start_time
    
    # Расчет статистик по всем сделкам
    if results['trades']:
        import pandas as pd
        import numpy as np
        
        results['stats'] = {
            'total_trades': len(results['trades']),
            'win_trades': sum(1 for t in results['trades'] if t['pnl'] > 0),
            'loss_trades': sum(1 for t in results['trades'] if t['pnl'] <= 0),
            'win_rate': sum(1 for t in results['trades'] if t['pnl'] > 0) / len(results['trades']),
            'avg_trade_pnl': np.mean([t['pnl'] for t in results['trades']]),
            'avg_win_pnl': np.mean([t['pnl'] for t in results['trades'] if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in results['trades']) else 0,
            'avg_loss_pnl': np.mean([t['pnl'] for t in results['trades'] if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in results['trades']) else 0,
            'max_win_pnl': max([t['pnl'] for t in results['trades']], default=0),
            'max_loss_pnl': min([t['pnl'] for t in results['trades']], default=0),
            'avg_commission': np.mean([t['commission'] for t in results['trades']]),
            'avg_funding': np.mean([t['funding'] for t in results['trades']]),
            'total_commission': sum(t['commission'] for t in results['trades']),
            'total_funding': sum(t['funding'] for t in results['trades']),
            'avg_holding_hours': np.mean([t['holding_hours'] for t in results['trades']]),
            'max_concurrent_open': params.max_concurrent_trades
        }
        
        # Если доступна информация о проскальзывании, добавляем её в статистику
        if all('slippage' in t and 'total_cost' in t['slippage'] for t in results['trades']):
            results['stats']['total_slippage_cost'] = sum(t['slippage']['total_cost'] for t in results['trades'])
            results['stats']['avg_slippage_cost'] = np.mean([t['slippage']['total_cost'] for t in results['trades']])
            results['stats']['slippage_pct_of_pnl'] = abs(results['stats']['total_slippage_cost'] / results['total_pnl']) if results['total_pnl'] != 0 else 0
        
        # Добавление дополнительной информации по типам выхода из позиции
        exit_reasons = {t['exit_reason'] for t in results['trades'] if 'exit_reason' in t}
        for reason in exit_reasons:
            reason_count = sum(1 for t in results['trades'] if t.get('exit_reason') == reason)
            results['stats'][f'exits_{reason}'] = reason_count
            results['stats'][f'exits_{reason}_pct'] = reason_count / len(results['trades'])

    return results
