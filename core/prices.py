"""
Модуль для работы с временными рядами цен.
Содержит функции для получения, обработки и анализа ценовых данных.
"""

import logging
from typing import Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

# Глобальная переменная для доступа к датасету
from core.data import DATASET

def safe_ffill_with_gap_limit(series: pd.Series, max_gap_minutes: int = 60) -> pd.Series:
    """
    Безопасное заполнение пропусков с ограничением максимального гэпа.
    
    Args:
        series (pd.Series): Временной ряд для заполнения
        max_gap_minutes (int): Максимальный допустимый гэп в минутах
        
    Returns:
        pd.Series: Серия с заполненными пропусками (только для коротких гэпов)
    """
    if series.empty:
        return series
    
    # Сначала делаем обычный ffill
    filled_series = series.ffill()
    
    # Находим позиции с валидными данными в оригинальной серии
    valid_mask = series.notna()
    
    # Если все значения валидны, возвращаем заполненную серию
    if valid_mask.all():
        return filled_series
    
    # Создаем копию результата
    result = filled_series.copy()
    
    # Проходим по всем индексам и проверяем гэпы
    for i in range(len(series)):
        if not valid_mask.iloc[i]:  # Если это заполненное значение
            # Находим ближайший предыдущий валидный индекс
            prev_valid_idx = None
            for j in range(i-1, -1, -1):
                if valid_mask.iloc[j]:
                    prev_valid_idx = j
                    break
            
            if prev_valid_idx is not None:
                # Рассчитываем временной гэп
                current_time = series.index[i]
                last_valid_time = series.index[prev_valid_idx]
                gap_minutes = (current_time - last_valid_time).total_seconds() / 60
                
                # Если гэп больше допустимого, ставим NaN
                if gap_minutes > max_gap_minutes:
                    result.iloc[i] = np.nan
    
    return result

def fetch_series(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Извлекает временной ряд для указанного символа из датасета.
    
    Args:
        symbol (str): Символ для извлечения данных.
        start_ms (int): Начальная точка диапазона в миллисекундах Unix timestamp.
        end_ms (int): Конечная точка диапазона в миллисекундах Unix timestamp.
        
    Returns:
        pd.DataFrame | None: DataFrame с колонками ['ts_ms', 'close'] или None при ошибке.
    """
    if DATASET is None:
        logging.error(f"DATASET не инициализирован. Невозможно извлечь данные для {symbol}.")
        return None
    
    try:
        # Фильтр по символу и временному диапазону
        filter_expr = (
            (pc.field("symbol") == pc.scalar(symbol)) &
            (pc.field("ts_ms") >= pc.scalar(start_ms)) &
            (pc.field("ts_ms") < pc.scalar(end_ms))
        )
        
        # Запрашиваем только нужные колонки
        table = DATASET.to_table(
            filter=filter_expr,
            columns=["ts_ms", "close"]
        )
        
        # Преобразуем pyarrow table в pandas dataframe
        df = table.to_pandas()
        
        if df.empty:
            # logging.debug(f"Пустой DataFrame для {symbol} в диапазоне {start_ms} - {end_ms}")
            return None
            
        # Преобразование ts_ms в datetime индекс
        df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
        df = df.set_index('datetime')
        
        return df
        
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных для {symbol}: {e}")
        return None

def get_log_prices(pair: tuple[str, str], start_dt: datetime, end_dt: datetime, 
                  max_gap_minutes: int = 60) -> tuple[np.ndarray, pd.DatetimeIndex] | None:
    """
    Получает логарифмированные цены для пары символов в заданном временном диапазоне.

    Args:
        pair (tuple[str, str]): Кортеж из двух символов (например, ('BTCUSDT', 'ETHUSDT')).
        start_dt (datetime): Начальная дата и время диапазона.
        end_dt (datetime): Конечная дата и время диапазона (не включая эту точку).
        max_gap_minutes (int): Максимальный допустимый гэп в минутах для заполнения.

    Returns:
        tuple[np.ndarray, pd.DatetimeIndex] | None: 
            Кортеж из двумерного NumPy массива [2xN] с логарифмированными ценами
            (первая строка - первый символ, вторая - второй) и соответствующего pd.DatetimeIndex.
            Или None, если данные для одного из символов отсутствуют, недостаточны или возникла ошибка.
    """
    symbol1, symbol2 = pair

    # Преобразование datetime в миллисекунды Unix timestamp
    # Убедимся, что datetime объекты являются timezone-aware (UTC)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
        
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # a) Получение двух серий через fetch_series
    series1_df = fetch_series(symbol1, start_ms, end_ms)
    series2_df = fetch_series(symbol2, start_ms, end_ms)

    if series1_df is None or series1_df.empty:
        # logging.debug(f"Нет данных для {symbol1} в диапазоне {start_dt} - {end_dt} при вызове get_log_prices")
        return None
    if series2_df is None or series2_df.empty:
        # logging.debug(f"Нет данных для {symbol2} в диапазоне {start_dt} - {end_dt} при вызове get_log_prices")
        return None

    # b) Построение единого индекса с частотой 15 минут
    # Важно: pd.date_range генерирует ряд [start, end], поэтому для <end_dt используем end_dt - epsilon
    # Убедимся, что start_dt < end_dt, иначе date_range может вернуть пустой или некорректный результат
    if start_dt >= end_dt:
        logging.warning(f"Начальная дата {start_dt} больше или равна конечной дате {end_dt} для пары {pair}. Возвращен None.")
        return None
    
    # b) Построение единого индекса с частотой 15 минут
    # inclusive='left' чтобы end_dt не вошел, если он точно на границе 15-минутного интервала
    full_15T_index = pd.date_range(start=start_dt, end=end_dt, freq='15T', tz='UTC', inclusive='left')

    if full_15T_index.empty:
        logging.warning(f"Созданный 15-минутный индекс для {pair} в диапазоне {start_dt} - {end_dt} пуст. Проверьте даты.")
        return None

    # c) Переиндексация на полный 15-минутный индекс с безопасным заполнением пропусков
    # Используем новую функцию safe_ffill_with_gap_limit вместо обычного ffill
    series1_reindexed_raw = series1_df.reindex(full_15T_index)
    series2_reindexed_raw = series2_df.reindex(full_15T_index)
    
    # Применяем безопасное заполнение с ограничением гэпа
    series1_reindexed = pd.DataFrame({
        'close': safe_ffill_with_gap_limit(series1_reindexed_raw['close'], max_gap_minutes)
    })
    series2_reindexed = pd.DataFrame({
        'close': safe_ffill_with_gap_limit(series2_reindexed_raw['close'], max_gap_minutes)
    })

    # Собираем DataFrame для логарифмирования
    # Колонки будут называться 'close' для обоих, что нормально для np.log
    # Но для ясности и избежания конфликтов, если бы мы сохраняли df, лучше переименовать
    # df_for_log = pd.concat([series1_reindexed.rename(columns={'close': symbol1}), 
    #                        series2_reindexed.rename(columns={'close': symbol2})], axis=1)
    
    # Проще взять .values напрямую, если колонки 'close' стандартные
    prices_array = np.vstack([
        series1_reindexed['close'].values,
        series2_reindexed['close'].values
    ]).astype(float)

    # Проверка на наличие только NaN перед логарифмированием
    if np.isnan(prices_array).all():
        # logging.debug(f"Все значения цен для {pair} являются NaN после reindex на общий индекс.")
        return None

    log_prices_array = np.log(prices_array)
    
    # Проверка на наличие только NaN после логарифмирования (например, если цены были <= 0)
    if np.isnan(log_prices_array).all():
        # logging.debug(f"Все значения логарифмированных цен для {pair} являются NaN.")
        return None

    return log_prices_array, full_15T_index

# Пример тестирования safe_ffill_with_gap_limit (для отладки)
# if __name__ == "__main__":
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime, timedelta
    
#     # Создаем тестовую серию с пропусками
#     dates = pd.date_range(start='2023-01-01', end='2023-01-01 05:00:00', freq='15T')
#     values = [1.0, 2.0, np.nan, np.nan, 5.0, np.nan, 7.0, np.nan, np.nan, np.nan, np.nan, 12.0, 13.0, 14.0, np.nan, 16.0, np.nan, np.nan, np.nan, np.nan, 21.0]
#     test_series = pd.Series(values, index=dates)
    
#     print("Оригинальная серия:")
#     print(test_series)
    
#     print("\nОбычный ffill:")
#     print(test_series.ffill())
    
#     print("\nБезопасный ffill с лимитом 30 минут:")
#     safe_result = safe_ffill_with_gap_limit(test_series, max_gap_minutes=30)
#     print(safe_result)
    
#     print("\nБезопасный ffill с лимитом 60 минут:")
#     safe_result_60 = safe_ffill_with_gap_limit(test_series, max_gap_minutes=60)
#     print(safe_result_60)
