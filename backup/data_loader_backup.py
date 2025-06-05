#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Импорты
import os
import sys
import time
import json
import asyncio
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from functools import lru_cache

import numpy as np
import pandas as pd
import httpx
from httpx import Limits, AsyncClient
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import duckdb

# Константы
BASE_DIR = "data"
START_TS = pd.Timestamp("2022-01-01", tz="UTC")
END_TS = pd.Timestamp("2025-06-01", tz="UTC")
TF_MIN = 15
API_RATE_LIMIT = 120
PARALLEL_NET = 120
PARALLEL_DISK = os.cpu_count()

# Основные функции
def load_symbols():
    """Загружает список символов из файла Markets.txt"""
    try:
        with open("Markets.txt", "r") as f:
            content = f.read()
        return [s.strip() for s in content.split(",") if s.strip()]
    except Exception as e:
        logger.error(f"Ошибка при загрузке символов: {e}")
        return []

def ensure_dirs():
    """Создает необходимые директории для хранения данных"""
    Path(BASE_DIR).mkdir(exist_ok=True)

def arrow_table(json_chunk, sym):
    """Преобразует JSON-данные в Arrow Table"""
    if not json_chunk:
        return None
    
    # Создаем Arrow массивы из данных
    ts_ms = pa.array([item['t'] for item in json_chunk], pa.timestamp('ms'))
    opens = pa.array([float(item['o']) for item in json_chunk], pa.float64())
    highs = pa.array([float(item['h']) for item in json_chunk], pa.float64())
    lows = pa.array([float(item['l']) for item in json_chunk], pa.float64())
    closes = pa.array([float(item['c']) for item in json_chunk], pa.float64())
    volumes = pa.array([float(item['v']) for item in json_chunk], pa.float64())
    symbols = pa.array([sym] * len(json_chunk), pa.string())
    
    # Создаем таблицу
    table = pa.Table.from_arrays(
        [symbols, opens, highs, lows, closes, volumes, ts_ms],
        names=["symbol", "open", "high", "low", "close", "volume", "ts_ms"]
    )
    
    return table

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_loader.log")
    ]
)
logger = logging.getLogger(__name__)

def write(table):
    """Записывает таблицу в партиционированный Parquet файл"""
    if table is None or len(table) == 0:
        return
    
    # Добавляем столбцы для партиционирования
    year = pc.year(table["ts_ms"])
    table = table.append_column("year", year)
    
    month = pc.month(table["ts_ms"])
    table = table.append_column("month", month)
    
    try:
        pq.write_to_dataset(
            table,
            BASE_DIR,
            partition_cols=["symbol", "year", "month"],
            existing_data_behavior="overwrite_or_ignore"
        )
    except Exception as e:
        logger.error(f"Ошибка при записи данных: {e}")

def gaps(sym):
    """Находит пропуски в данных для символа"""
    try:
        # Путь к данным символа
        sym_path = f"{BASE_DIR}/symbol={sym}"
        
        if not Path(sym_path).exists():
            # Если данных нет, возвращаем весь интервал
            start_ms = int(START_TS.timestamp() * 1000)
            end_ms = int(END_TS.timestamp() * 1000)
            return [(start_ms, end_ms)]
        
        # Загружаем существующие временные метки
        dataset = ds.dataset(sym_path)
        existing_df = dataset.to_table(columns=["ts_ms"]).to_pandas()
        existing = existing_df["ts_ms"].values
        
        # Создаем ожидаемые временные метки
        start_ms = int(START_TS.timestamp() * 1000)
        end_ms = int(END_TS.timestamp() * 1000)
        step_ms = TF_MIN * 60 * 1000
        expected = np.arange(start_ms, end_ms + 1, step_ms)
        
        # Находим отсутствующие метки
        missing = np.setdiff1d(expected, existing)
        
        # Группируем в непрерывные интервалы
        if len(missing) == 0:
            return []
            
        ranges = []
        start_range = missing[0]
        prev = missing[0]
        
        for ts in missing[1:]:
            if ts - prev > step_ms:
                ranges.append((start_range, prev))
                start_range = ts
            prev = ts
            
        ranges.append((start_range, prev))
        return ranges
        
    except Exception as e:
        logger.error(f"Ошибка при поиске пропусков для {sym}: {e}")
        # В случае ошибки возвращаем весь интервал
        start_ms = int(START_TS.timestamp() * 1000)
        end_ms = int(END_TS.timestamp() * 1000)
        return [(start_ms, end_ms)]

async def http_request(client, url, params):
    """Выполняет HTTP запрос с повторными попытками"""
    retries = 3
    backoff = 1.0
    
    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == retries:
                logger.error(f"Ошибка запроса {url}: {e}")
                return None
            await asyncio.sleep(backoff)
            backoff *= 2

async def fetch(sym, ranges, client, pool, semaphore):
    """Загружает данные для символа по заданным интервалам"""
    if not ranges:
        logger.info(f"[{sym}] Данные уже полные, пропускаем")
        return 0
    
    url = "https://api.binance.com/api/v3/klines"
    interval = f"{TF_MIN}m"
    total_rows = 0
    
    for start_ms, end_ms in ranges:
        start_str = pd.Timestamp(start_ms, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        end_str = pd.Timestamp(end_ms, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"[{sym}] Загрузка данных {start_str} - {end_str}")
        
        async with semaphore:
            params = {
                "symbol": sym,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000
            }
            
            data = await http_request(client, url, params)
            
            if data and len(data) > 0:
                # Преобразуем формат данных Binance в наш формат
                json_chunk = [
                    {
                        "t": item[0],  # timestamp
                        "o": item[1],  # open
                        "h": item[2],  # high
                        "l": item[3],  # low
                        "c": item[4],  # close
                        "v": item[5],  # volume
                    }
                    for item in data
                ]
                
                # Создаем Arrow таблицу
                table = arrow_table(json_chunk, sym)
                
                if table is not None:
                    # Записываем данные асинхронно
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(pool, write, table)
                    total_rows += len(table)
    
    logger.info(f"[{sym}] Загружено {total_rows} строк")
    return total_rows

def stats():
    """Возвращает статистику по загруженным данным"""
    try:
        # Создаем DuckDB соединение для анализа данных
        conn = duckdb.connect(":memory:")
        
        # Запрос для подсчета строк по символам
        query = f"""
        SELECT symbol, COUNT(*) as rows
        FROM parquet_scan('{BASE_DIR}/**/*.parquet')
        GROUP BY symbol
        ORDER BY rows DESC
        """
        
        result = conn.execute(query).fetchall()
        total = sum(row[1] for row in result)
        
        # Форматируем результат
        stats_str = "Статистика загруженных данных:\n"
        stats_str += f"Всего строк: {total:,}\n"
        stats_str += "Топ-10 символов по количеству строк:\n"
        
        for i, (symbol, count) in enumerate(result[:10], 1):
            stats_str += f"{i}. {symbol}: {count:,}\n"
            
        return stats_str
    except Exception as e:
        logger.error(f"Ошибка при расчете статистики: {e}")
        return "Не удалось получить статистику"

async def main(full_reload=False):
    """Основная функция загрузки данных"""
    start_time = time.time()
    
    # Создаем необходимые директории
    ensure_dirs()
    
    # Если указан полный перезапуск, удаляем существующие данные
    if full_reload:
        logger.info("Удаление существующих данных...")
        for path in Path(BASE_DIR).glob("**/*.parquet"):
            path.unlink()
    
    # Загружаем символы
    symbols = load_symbols()
    if not symbols:
        logger.error("Не найдены символы в Markets.txt")
        return
    
    logger.info(f"Загрузка данных для {len(symbols)} символов с {START_TS} по {END_TS}")
    
    # Создаем HTTP клиент и пул потоков для записи
    limits = Limits(max_connections=PARALLEL_NET)
    client = AsyncClient(limits=limits)
    pool = ThreadPoolExecutor(max_workers=PARALLEL_DISK)
    semaphore = asyncio.Semaphore(PARALLEL_NET)
    
    try:
        # Подготавливаем задачи для каждого символа
        tasks = []
        for sym in symbols:
            # Находим пропуски в данных, если не требуется полная перезагрузка
            if not full_reload:
                missing_ranges = gaps(sym)
            else:
                # При полной перезагрузке загружаем весь интервал
                start_ms = int(START_TS.timestamp() * 1000)
                end_ms = int(END_TS.timestamp() * 1000)
                missing_ranges = [(start_ms, end_ms)]
            
            task = fetch(sym, missing_ranges, client, pool, semaphore)
            tasks.append(task)
        
        # Запускаем все задачи параллельно
        results = await asyncio.gather(*tasks)
        total_rows = sum(results)
        
        # Выводим статистику
        logger.info(f"Загрузка завершена. Загружено {total_rows} строк")
        logger.info(f"Время выполнения: {time.time() - start_time:.2f} сек")
        logger.info(stats())
        
    finally:
        # Закрываем клиент и пул потоков
        await client.aclose()
        pool.shutdown()

if __name__ == "__main__":
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Загрузчик исторических данных")
    parser.add_argument("--full-reload", action="store_true", help="Полная перезагрузка данных")
    args = parser.parse_args()
    
    # Запускаем основную функцию
    asyncio.run(main(full_reload=args.full_reload))
    try:
        # Создаем временный файл для сохранения
        temp_file = HISTORICAL_DATA_FILE + '.temp'
        
        # Обрабатываем данные посимвольно для экономии памяти
        first_symbol = True
        total_rows = 0
        
        for symbol, data in cache.items():
            if not data:
                continue
                
            # Создаем DataFrame для текущего символа
            df = pd.DataFrame(data)
            
            # Оптимизируем типы данных
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'close' in df.columns:
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Добавляем символ как колонку
            df['symbol'] = symbol
            
            # Сохраняем в Parquet с добавлением данных
            if first_symbol:
                df.to_parquet(temp_file, compression='snappy', index=False)
                first_symbol = False
            else:
                df.to_parquet(temp_file, compression='snappy', index=False, append=True)
            
            total_rows += len(df)
            logger.info(f"Сохранено {len(df)} записей для символа {symbol}")
            
            # Очищаем память
            del df
        
        if total_rows > 0:
            # Переименовываем временный файл в финальный
            if os.path.exists(HISTORICAL_DATA_FILE):
                os.remove(HISTORICAL_DATA_FILE)
            os.rename(temp_file, HISTORICAL_DATA_FILE)
            logger.info(f"Кэш сохранен в {HISTORICAL_DATA_FILE} (всего {total_rows} записей)")
        else:
            logger.warning("Нет данных для сохранения")
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except Exception as e:
        logger.error(f"Ошибка при сохранении кэша: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def validate_date(date_str: str) -> bool:
    """
    Проверяет корректность даты.
    
    Args:
        date_str: Строка с датой в формате YYYY-MM-DD
        
    Returns:
        bool: True если дата корректна
    """
    try:
        pd.Timestamp(date_str)
        return True
    except:
        return False

@lru_cache(maxsize=1000)
def calculate_technical_indicators(prices: Tuple[float, ...], volumes: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Оптимизированный расчет технических индикаторов"""
    prices_arr = np.array(prices)
    volumes_arr = np.array(volumes)
    
    vwap = np.average(prices_arr, weights=volumes_arr)
    momentum = prices_arr[-1] - prices_arr[0] if len(prices_arr) > 1 else 0
    volatility = np.std(prices_arr) if len(prices_arr) > 1 else 0
    
    return vwap, momentum, volatility

def process_chunk_data(chunk_data: List[dict], symbol: str, chunk_id: int = 0) -> pd.DataFrame:
    """Обработка чанка данных в отдельном процессе с оптимизированными вычислениями"""
    try:
        # Преобразуем данные в numpy массивы
        data = np.array(chunk_data)
        
        # Создаем DataFrame с правильными типами данных
        df = pd.DataFrame({
            'timestamp': pd.to_numeric(data[:, 0], errors='coerce'),
            'open': pd.to_numeric(data[:, 1], errors='coerce'),
            'high': pd.to_numeric(data[:, 2], errors='coerce'),
            'low': pd.to_numeric(data[:, 3], errors='coerce'),
            'close': pd.to_numeric(data[:, 4], errors='coerce'),
            'volume': pd.to_numeric(data[:, 5], errors='coerce'),
            'turnover': pd.to_numeric(data[:, 6], errors='coerce')
        })
    
        # Вычисляем технические индикаторы
        df['vwap'] = df['turnover'] / df['volume']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_range'] = df['high'] - df['low']
        
        # Расчет скользящих индикаторов
        window_size = min(20, len(df))
        if window_size > 1:
            # Вычисляем моментум и волатильность
            df['momentum'] = df['close'].diff(window_size)
            df['volatility'] = df['close'].rolling(window=window_size).std()
            
            # Вычисляем ускорение цены
            df['price_acceleration'] = df['close'].diff().diff()
        else:
            # Если данных мало, заполняем NaN
            df['momentum'] = np.nan
            df['volatility'] = np.nan
            df['price_acceleration'] = np.nan
    
        # Преобразуем timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Добавляем символ
        df['symbol'] = symbol
        
        return df
        
    except Exception as e:
        logger.error(f"[Символ {symbol}] Ошибка при обработке чанка {chunk_id}: {str(e)}")
        logger.debug(f"[Символ {symbol}] Размер чанка: {len(chunk_data)}, Первая запись: {chunk_data[0] if chunk_data else 'None'}")
        return None
    
    # Преобразуем timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Добавляем символ
    df['symbol'] = symbol
    
    return df
    """Обработка чанка данных в отдельном процессе"""
    df = pd.DataFrame(chunk_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 
        'volume', 'turnover'
    ])
    
    # Добавляем символ
    df['symbol'] = symbol
    
    # Преобразуем типы
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

# Константы для работы с API
API_RATE_LIMIT = 120  # запросов в секунду (новое значение)
MIN_REQUEST_INTERVAL = 1.0 / API_RATE_LIMIT  # минимальный интервал между запросами
REQUEST_WINDOW = 1.0  # окно для подсчета RPS

# Настройки для HTTP/2
HTTP2_ENABLED = True
CHUNK_SIZE = 500  # Увеличиваем размер чанка
MAX_CONCURRENT_REQUESTS = 120  # Максимальное количество одновременных запросов (новое значение)

# Настройки для обработки ошибок
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 5.0

# Счетчики запросов
class RequestTracker:
    def __init__(self):
        self.requests = []
        self.total_requests = 0
        self.start_time = time.time()
    
    def add_request(self):
        current_time = time.time()
        self.requests.append(current_time)
        self.total_requests += 1
        
        # Удаляем старые запросы
        while self.requests and current_time - self.requests[0] > REQUEST_WINDOW:
            self.requests.pop(0)
    
    def get_current_rps(self) -> float:
        current_time = time.time()
        return len(self.requests) / REQUEST_WINDOW
    
    def get_average_rps(self) -> float:
        current_time = time.time()
        elapsed = current_time - self.start_time
        return self.total_requests / elapsed if elapsed > 0 else 0

# Создаем глобальный трекер запросов
request_tracker = RequestTracker()
session_last_request = {}

def get_delay_time(session: HTTP) -> float:
    """Рассчитывает необходимую задержку для соблюдения лимитов и логирует ее."""
    current_time = time.time()
    last_request_time = session_last_request.get(id(session), 0)
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < MIN_REQUEST_INTERVAL:
        delay = MIN_REQUEST_INTERVAL - time_since_last_request
        logger.info(f"Задержка {delay:.3f} секунд перед выполнением запроса (time_since_last_request: {time_since_last_request:.3f})")
        return delay
    return 0

def fetch_symbol_data(
    session: HTTP,
    symbol: str,
    start_time: str,
    end_time: str,
    chunk_size: int = 200,
    retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0
) -> Optional[pd.DataFrame]:
    logger.info(f"Начало загрузки данных для символа {symbol} за период {start_time} - {end_time}")
    # Проверяем корректность дат
    if not validate_date(start_time) or not validate_date(end_time):
        logger.error(f"Некорректный формат даты: {start_time} - {end_time}")
        return None
        
    start_ts = int(pd.Timestamp(start_time).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_time).timestamp() * 1000)
    
    if start_ts >= end_ts:
        logger.error(f"Начальная дата {start_time} позже или равна конечной {end_time}")
        return None
    
    # Вычисляем и выводим текущее время задержки перед загрузкой для данной пары
    current_delay = get_delay_time(session)
    logger.info(f"Текущее время задержки перед загрузкой для {symbol}: {current_delay:.3f} секунд")
    if current_delay > 0:
        time.sleep(current_delay)
    
    delay = initial_delay
    
    for attempt in range(retries):
        try:
            # Разбиваем период на чанки по chunk_size таймстемпов
            all_data = []
            current_start = start_ts
            
            total_chunks = (end_ts - start_ts) // (chunk_size * 15 * 60 * 1000) + 1
            current_chunk = 0
            
            logger.info(f"[Символ {symbol}] Всего чанков для загрузки: {total_chunks}")
            
            while current_start < end_ts:
                current_chunk += 1
                current_end = min(current_start + (chunk_size * 15 * 60 * 1000), end_ts)
                
                logger.debug(f"[{symbol}] Загрузка чанка {current_chunk}/{total_chunks}")
                
                response = session.get_kline(
                    category="spot",
                    symbol=symbol,
                    interval="15",  # 15 минутный интервал
                    start=current_start,
                    end=current_end,
                    limit=chunk_size
                )
            
                if response['retCode'] != 0:
                    raise ValueError(f"API вернул ошибку: {response['retMsg']}")
                    
                data = response['result']['list']
                if data:
                    all_data.extend(data)
                    
                # Обновляем статистику запросов
                request_tracker.add_request()
                current_rps = request_tracker.get_current_rps()
                
                # Динамическая задержка в зависимости от RPS
                if current_rps > API_RATE_LIMIT:
                    excess_rps = current_rps - API_RATE_LIMIT
                    delay = excess_rps / (API_RATE_LIMIT * 10)
                    if delay > 0:
                        time.sleep(delay)
                
                if request_tracker.total_requests % 50 == 0:
                    logger.info(f"Статистика запросов: RPS={current_rps:.1f}, Средний RPS={request_tracker.get_average_rps():.1f}")
                
                delay = get_rate_limit_delay(current_rps)
                if delay > 0:
                    time.sleep(delay)
                session_last_request[id(session)] = time.time()
                
                current_start = current_end
            
            if not all_data:
                logger.warning(f"[{symbol}] Нет данных за период {start_time} - {end_time}")
                return None
            
            logger.info(f"[{symbol}] Успешно загружено {len(all_data)} записей")
            
            chunk_size = max(1000, len(all_data) // multiprocessing.cpu_count())
            data_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
            
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as process_executor:
                chunk_futures = [
                    process_executor.submit(process_chunk_data, chunk, symbol, i)
                    for i, chunk in enumerate(data_chunks)
                ]
                
                processed_chunks = []
                for future in as_completed(chunk_futures):
                    try:
                        chunk_df = future.result(timeout=30)
                        if chunk_df is not None:
                            processed_chunks.append(chunk_df)
                    except Exception as e:
                        logger.error(f"[{symbol}] Ошибка при обработке чанка данных: {str(e)}")
                
                if not processed_chunks:
                    logger.error(f"[{symbol}] Не удалось обработать ни один чанк данных")
                    return None
                
                try:
                    df = pd.concat(processed_chunks, ignore_index=True)
                    logger.info(f"[{symbol}] Данные успешно преобразованы в DataFrame ({len(df)} записей)")
                    return df
                except Exception as e:
                    logger.error(f"[{symbol}] Ошибка при объединении чанков: {str(e)}")
                    return None
            
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"[{symbol}] Попытка {attempt + 1}/{retries} не удалась: {str(e)}")
                logger.info(f"[{symbol}] Ожидание {delay:.1f} секунд перед следующей попыткой...")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"[{symbol}] Все попытки ({retries}) исчерпаны: {str(e)}")
                return None
                
    logger.error(f"Критическая ошибка при получении данных для {symbol}")
    return None

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        try:
            # Профилирование функции
            profiled_func = profiler(func)
            result = profiled_func(*args, **kwargs)
            
            # Сохраняем результаты профилирования
            s = io.StringIO()
            profiler.print_stats(stream=s)
            logging.info(f"Profile results for {func.__name__}:\n{s.getvalue()}")
            
            return result
        finally:
            profiler.disable()
    return wrapper

@profile_function
async def make_request(session: aiohttp.ClientSession, url: str, params: dict) -> Optional[dict]:
    """Выполнение HTTP запроса с обработкой ошибок и ретраем"""
    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (compatible; BybitBot/1.0)',
                'Accept-Encoding': 'gzip, deflate, br',
            }
            
            async with session.get(url, params=params, headers=headers, ssl=False) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Экспоненциальная задержка
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    logger.warning(f"Превышен лимит API, ожидание {delay} секунд")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Ошибка API {response.status}: {await response.text()}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(BASE_DELAY)
                        continue
                    return None
                    
        except Exception as e:
            logger.error(f"Ошибка запроса: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_DELAY)
                continue
            return None
    
    return None

async def fetch_symbol_data_async(
    session: aiohttp.ClientSession,
    symbol: str,
    start_time: str,
    end_time: str,
    chunk_size: int = CHUNK_SIZE
) -> Optional[pd.DataFrame]:
    """Асинхронная загрузка данных для одного символа"""
    try:
        # Преобразование времени в timestamp
        start_ts = int(pd.Timestamp(start_time).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_time).timestamp() * 1000)
        
        # Разбиваем период на чанки
        time_chunks = []
        current_start = start_ts
        while current_start < end_ts:
            current_end = min(current_start + (chunk_size * 60000), end_ts)
            time_chunks.append((current_start, current_end))
            current_start = current_end
        
        # Создаем задачи для каждого чанка
        tasks = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def fetch_chunk(start: int, end: int):
            async with semaphore:
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'interval': '1',
                    'start': start,
                    'end': end,
                    'limit': chunk_size
                }
                return await make_request(
                    session=session,
                    url='https://api.bybit.com/v5/market/kline',
                    params=params
                )
        
        tasks = [fetch_chunk(start, end) for start, end in time_chunks]
        
        # Запускаем все запросы параллельно
        all_data = []
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            if response is None:
                continue
            
            data = response.get('result', {}).get('list', [])
            if data:
                all_data.extend(data)
        
        if not all_data:
            return None
        
        # Преобразование в DataFrame
        df = pd.DataFrame(all_data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        # Преобразование типов
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        df['symbol'] = symbol
        
        return df
    
    except Exception as e:
        logger.error(f"[Символ {symbol}] Ошибка: {str(e)}")
        return None

async def fetch_symbols_async(symbols: List[str], start_time: str, end_time: str) -> Dict[str, pd.DataFrame]:
    """Асинхронная загрузка данных для всех символов"""
    try:
        # Создаем клиент Bybit
        session = HTTP(
            testnet=False,
            api_key=None,
            api_secret=None,
            recv_window=5000
        )
        
        tasks = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        results = {}
        processed_count = 0
        batch_size = 15
        
        def save_batch_to_file(batch_results: Dict[str, pd.DataFrame], batch_num: int):
            logger.info(f"Начало сохранения партии {batch_num} для {len(batch_results)} валют.")
            try:
                # Преобразуем словарь DataFrame'ов в один DataFrame
                all_data = []
                for symbol, df in batch_results.items():
                    # Сбрасываем индекс и добавляем символ
                    df_reset = df.reset_index()
                    df_reset['symbol'] = symbol
                    # Приводим числовые колонки к float
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        df_reset[col] = df_reset[col].astype(float)
                    all_data.append(df_reset)
                
                new_data = pd.concat(all_data, ignore_index=True)
                
                # Создаем директорию, если она не существует
                directory = os.path.dirname(HISTORICAL_DATA_FILE)
                if not os.path.exists(directory):
                    logger.info(f"Создаем директорию: {directory}")
                    os.makedirs(directory, exist_ok=True)
                
                logger.info(f"Проверяем существование файла: {os.path.abspath(HISTORICAL_DATA_FILE)}")
                # Если файл существует, дописываем новые данные с использованием fastparquet и параметра append
                if os.path.exists(HISTORICAL_DATA_FILE):
                    logger.info(f"Файл существует, добавляем данные")
                    new_data.to_parquet(HISTORICAL_DATA_FILE, engine='fastparquet', append=True, compression='snappy', index=False)
                    logger.info(f"Партия {batch_num}: добавлено {len(new_data)} записей в {HISTORICAL_DATA_FILE}")
                else:
                    logger.info(f"Файл не существует, создаем новый")
                    new_data.to_parquet(HISTORICAL_DATA_FILE, engine='fastparquet', compression='snappy', index=False)
                    logger.info(f"Партия {batch_num}: создан файл {HISTORICAL_DATA_FILE} с {len(new_data)} записями")
                
                # Проверяем, что файл сохранился
                if os.path.exists(HISTORICAL_DATA_FILE):
                    logger.info(f"Файл успешно сохранен, размер: {os.path.getsize(HISTORICAL_DATA_FILE)} байт")
                else:
                    logger.error(f"Ошибка: файл не был создан")
                    
                return HISTORICAL_DATA_FILE
                
            except Exception as e:
                logger.error(f"Ошибка при сохранении партии {batch_num}: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return None
        
        async def fetch_with_semaphore(symbol: str):
            async with semaphore:
                try:
                    response = session.get_kline(
                        category="spot",
                        symbol=symbol,
                        interval="1",
                        start=start_time,
                        end=end_time,
                        limit=1000
                    )
                    
                    if response and 'result' in response and 'list' in response['result']:
                        data = response['result']['list']
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        return df
                    else:
                        logger.error(f"[Символ {symbol}] Некорректный ответ API")
                        return None
                        
                except (FailedRequestError, InvalidRequestError) as e:
                    logger.error(f"[Символ {symbol}] Ошибка API: {str(e)}")
                    return None
                except Exception as e:
                    logger.error(f"[Символ {symbol}] Неожиданная ошибка: {str(e)}")
                    return None
        
        # Создаем задачи для каждого символа
        tasks = [(symbol, asyncio.create_task(fetch_with_semaphore(symbol))) for symbol in symbols]
        
        # Ждем завершения всех задач
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Обрабатываем результаты
        batch_results = {}
        saved_files = []
        
        for (symbol, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"[Символ {symbol}] Ошибка: {str(result)}")
                continue
            
            if result is not None:
                # Преобразуем данные в DataFrame
                df = pd.DataFrame(result)
                if not df.empty:
                    results[symbol] = result
                    batch_results[symbol] = df  # Сохраняем DataFrame вместо сырых данных
                    processed_count += 1
                    logger.info(f"[Символ {symbol}] Успешно загружены данные ({len(df)} записей)")
                else:
                    logger.warning(f"[Символ {symbol}] Получены пустые данные")
                
                # Сохраняем каждые 15 символов
                if processed_count % batch_size == 0:
                    batch_num = processed_count // batch_size
                    try:
                        filename = save_batch_to_file(batch_results, batch_num)
                        if filename and os.path.exists(filename):
                            logger.info(f"Успешно сохранена партия {batch_num} ({len(batch_results)} символов)")
                            saved_files.append(filename)
                            batch_results = {}
                        else:
                            logger.error(f"Не удалось сохранить партию {batch_num} - файл не создан")
                    except Exception as e:
                        logger.error(f"Ошибка при сохранении партии {batch_num}: {str(e)}")
        
        # Сохраняем оставшиеся данные, если есть
        if batch_results:
            batch_num = (processed_count // batch_size) + 1
            filename = save_batch_to_file(batch_results, batch_num)
            if filename:
                saved_files.append(filename)
        
        logger.info(f"Всего сохранено файлов: {len(saved_files)}")
        return results
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return {}

def fetch_all_data_parallel(symbols: List[str], total_days_to_load: Optional[int] = None, max_workers: Optional[int] = None) -> Tuple[Dict[str, List], bool]:
    """Параллельная загрузка данных с использованием асинхронных запросов"""
    try:
        # Инициализация переменных
        start_time = calculate_start_time(total_days_to_load)
        end_time = calculate_end_time()
        
        logger.info(f"Загрузка данных с {start_time} по {end_time}")
        
        # Разбиваем символы на батчи для оптимальной загрузки
        batch_size = MAX_CONCURRENT_REQUESTS  # Размер батча равен максимальному количеству параллельных запросов
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        # Создаем общий event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Запускаем все батчи параллельно
            all_results = {}
            for i, batch in enumerate(symbol_batches, 1):
                logger.info(f"Загрузка батча {i}/{len(symbol_batches)} (символов: {len(batch)})")
                batch_results = loop.run_until_complete(fetch_symbols_async(batch, start_time, end_time))
                all_results.update(batch_results)
                
                # Небольшая пауза между батчами для предотвращения перегрузки
                if i < len(symbol_batches):
                    loop.run_until_complete(asyncio.sleep(0.1))
            
            # Проверка результатов
            success = len(all_results) > 0
            if success:
                logger.info(f"Успешно загружены данные для {len(all_results)} из {len(symbols)} символов")
            else:
                logger.error("Не удалось загрузить данные ни для одного символа")
            
            return all_results, success
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return {}, False
        
        # Проверка результатов
        success = len(all_results) > 0
        if success:
            logger.info(f"Успешно загружены данные для {len(all_results)} из {len(symbols)} символов")
        else:
            logger.error("Не удалось загрузить данные ни для одного символа")
        
        return all_results, success
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return {}, False
    
    # Оптимальное количество сессий и воркеров
    if not max_workers:
        max_workers = min(10, os.cpu_count() * 2)  # Оптимальное количество сессий
    
    logger.info(f"Запуск загрузки данных с целевым RPS={API_RATE_LIMIT}, воркеров={max_workers}")
    """Параллельная загрузка и обработка данных для всех символов.
    
    Оптимизировано для максимальной утилизации CPU с использованием:
    - Агрессивной предварительной загрузки данных
    - Параллельной обработки на уровне чанков
    - Оптимизированных numpy операций
    - Динамической балансировки нагрузки
    """
    """Параллельная загрузка и обработка данных для всех символов.
    
    Использует комбинацию ThreadPoolExecutor для I/O операций и
    ProcessPoolExecutor для обработки данных.
    """
    logger.info(f"Начинаем параллельную загрузку данных для {len(symbols)} символов")
    """
    Параллельная загрузка данных для всех символов.
    
    Args:
        symbols: Список символов для загрузки
        total_days_to_load: Количество дней для загрузки
        
    Returns:
        Tuple[Dict, bool]: (словарь с данными, флаг изменения данных)
    """
    try:
        # Создаем директорию для логов если её нет
        log_dir = os.path.dirname('data_loader.log')
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Инициализация переменных должна быть до создания session_cycle
        _sessions = []
        cache = load_cached_data()
        all_data = []
        _batch_results = {}
        _cached_batch = {}
        batch_size = 15
        processed_count = 0

        # Затем создание сессий и session_cycle
        for key, secret in zip(MAINNET_API_KEYS, MAINNET_API_SECRETS):
            try:
                session = HTTP(api_key=key, api_secret=secret)
                _sessions.append(session)
            except Exception as e:
                logger.error(f"Ошибка создания сессии: {str(e)}")

        if not _sessions:
            raise ValueError("Не удалось создать ни одной сессии API")

        session_cycle = cycle(_sessions)
        
        # Определяем временной интервал
        end_time = datetime.now()
        if total_days_to_load is None:
            total_days_to_load = 30  # По умолчанию загружаем за 30 дней
        logger.info(f"Установлен период загрузки: {total_days_to_load} дней")
            
        start_time = end_time - timedelta(days=total_days_to_load)
        
        # Форматируем даты
        start_str = start_time.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')
        
        # Распределяем символы по сессиям
        symbol_chunks = np.array_split(symbols, len(sessions))
        
        all_data = {}
        failed_symbols = set()
        data_changed = False
        
        # Добавляем ограничение по памяти
        MAX_MEMORY_USAGE = 1024 * 1024 * 1024  # 1GB
        
        def check_memory_usage() -> bool:
            try:
                process = psutil.Process()
                current_usage = process.memory_info().rss
                if current_usage > MAX_MEMORY_USAGE:
                    logger.warning(f"Достигнут предел использования памяти: {current_usage/1024/1024:.1f}MB")
                    logger.info("Начало промежуточной очистки памяти (gc.collect)")
                    gc.collect()
                    logger.info("Промежуточная очистка памяти завершена")
                    return False
                return True
            except Exception as e:
                logger.error(f"Ошибка при проверке памяти: {str(e)}")
                return False
        
        # Загружаем данные параллельно
        with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
            futures = []
            
            # Распределяем символы по воркерам
            symbols_per_worker = len(symbols) // max_workers
            symbol_batches = [
                symbols[i:i + symbols_per_worker] 
                for i in range(0, len(symbols), symbols_per_worker)
            ]
            
            # Предварительная загрузка первого батча
            prefetch_batch = symbol_batches[1] if len(symbol_batches) > 1 else []
            
            if prefetch_batch:
                logger.info(f"Предварительная загрузка следующего батча из {len(prefetch_batch)} символов")
                for symbol in prefetch_batch:
                    session = get_next_session(session_cycle)
                    if session is None:
                        logger.error(f"Нет доступной сессии для символа {symbol}")
                        continue
                    future = executor.submit(
                        fetch_symbol_data,
                        session=session,
                        symbol=symbol,
                        start_time=start_str,
                        end_time=end_str
                    )
                    futures.append((symbol, future))
            
            logger.info(f"Разбивка на {len(symbol_batches)} пакетов по {symbols_per_worker} символов")
            
            for batch in symbol_batches:
                for symbol in batch:
                    session = get_next_session(session_cycle)
                    if session is None:
                        logger.error(f"Нет доступной сессии для символа {symbol}")
                        continue
                    future = executor.submit(
                        fetch_symbol_data,
                        session=session,
                        symbol=symbol,
                        start_time=start_str,
                        end_time=end_str
                    )
                    futures.append((symbol, future))
            
            total_symbols = len(futures)
            processed_symbols = 0
            successful_symbols = 0
            logger.info(f"Начинаем обработку результатов для {total_symbols} символов...")
            
            # Собираем результаты
            for symbol, future in futures:
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_data[symbol] = df.to_dict('records')
                        successful_symbols += 1
                        logger.info(f"Успешно загружены данные для {symbol} ({len(df)} записей)")
                    else:
                        failed_symbols.add(symbol)
                except Exception as e:
                    logger.error(f"Ошибка при получении данных для {symbol}: {str(e)}")
                    failed_symbols.add(symbol)
        
        # Обновляем кэш
        for symbol, data in all_data.items():
            # Нельзя напрямую сравнивать DataFrame с другими данными
            if symbol not in cache:
                # Если символа нет в кэше, добавляем
                logger.info(f"Добавление нового символа {symbol} в кэш")
                # Преобразуем данные в DataFrame, если они еще не в этом формате
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    if 'timestamp' in df.columns:
                        df['timestamp_ms'] = pd.to_datetime(df['timestamp'], utc=True).astype('int64') // 1_000_000
                    cache[symbol] = df
                else:
                    cache[symbol] = data
                data_changed = True
                
        # Удаляем символы с ошибками из кэша
        for symbol in failed_symbols:
            if symbol in cache:
                del cache[symbol]
                data_changed = True
                
        # Сохраняем обновленный кэш только если были изменения
        if data_changed:
            logger.info(f"Сохранение обновленного кэша для {len(cache)} символов...")
            save_cached_data(cache)
            logger.info("Кэш успешно обновлен")
        else:
            logger.info("Изменений в кэше нет")
        
        logger.info(f"Итоги загрузки:")
        logger.info(f"- Всего символов: {total_symbols}")
        logger.info(f"- Успешно загружено: {successful_symbols}")
        logger.info(f"- Ошибок загрузки: {len(failed_symbols)}")
            
        logger.info("Завершена параллельная загрузка данных")
        if data_changed:
            logger.info("Обнаружены новые данные")
        else:
            logger.info("Новых данных не обнаружено")
        return cache, data_changed
        
    except Exception as e:
        logger.error(f"Ошибка при параллельной загрузке данных: {str(e)}")
        return {}, False

def get_parquet_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Получение информации о Parquet-файле.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Dict[str, Any]: Информация о файле
    """
    try:
        df = pd.read_parquet(file_path)
        return {
            'num_rows': len(df),
            'columns': df.columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'file_size': os.path.getsize(file_path)
        }
    except Exception as e:
        logger.error(f"Ошибка при получении информации о файле: {str(e)}")
        return {}

def has_data(data: List[dict]) -> bool:
    """
    Проверяет наличие данных.
    
    Args:
        data: Список словарей с данными
        
    Returns:
        bool: True если данные есть, False иначе
    """
    return bool(data)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Расширенная проверка корректности DataFrame"""
    try:
        if df is None:
            return False
        required_columns = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        if not all(col in df.columns for col in required_columns):
            return False
            
        # Проверяем типы данных
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            return False
            
        numeric_columns = {'open', 'high', 'low', 'close', 'volume'}
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False
                
        # Проверяем диапазон дат
        if df['timestamp'].min() > df['timestamp'].max():
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при валидации DataFrame: {str(e)}")
        return False

# Глобальные переменные для хранения состояния
_sessions = []
_batch_results = {}
_cached_batch = {}

def get_next_session(session_cycle) -> Optional[HTTP]:
    if session_cycle is None:
        logger.error("Не инициализирован цикл сессий")
        return None
    try:
        return next(session_cycle)
    except StopIteration:
        logger.error("Нет доступных сессий")
        return None

def collect_historical_data(symbols: List[str], 
                          start_date: str, end_date: str,
                          chunk_size: int = 200,
                          progress_interval: int = 10) -> pd.DataFrame:
    """Сбор исторических данных для списка символов."""
    global _sessions, _batch_results, _cached_batch
    global get_next_session
    session_getter = get_next_session
    try:
        # Проверяем даты
        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC')
        if start_ts >= end_ts:
            raise ValueError(f"Некорректный диапазон дат: {start_date} - {end_date}")
        
        # Инициализация сессий
        _sessions = []
        for key, secret in zip(MAINNET_API_KEYS, MAINNET_API_SECRETS):
            try:
                session = HTTP(api_key=key, api_secret=secret)
                _sessions.append(session)
            except Exception as e:
                logger.error(f"Ошибка создания сессии: {str(e)}")
        
        if not _sessions:
            raise ValueError("Не удалось создать ни одной сессии API")
        
        session_cycle = cycle(_sessions)
        
        # Инициализация переменных
        cache = load_cached_data()
        all_data = []
        _batch_results = {}
        _cached_batch = {}
        batch_size = 15
        processed_count = 0

        # Вспомогательная функция для поиска недостающих интервалов
        def find_missing_intervals(timestamps: List[int], start_ts: int, end_ts: int, interval_ms: int) -> List[Tuple[int, int]]:
            logger.info(f"Поиск недостающих интервалов (timestamp: {len(timestamps)}, диапазон: {start_ts}-{end_ts})")
            # Вычисляем все ожидаемые timestamp с шагом interval
            expected = list(range(start_ts, end_ts + 1, interval_ms))
            existing_set = set(timestamps)
            # Находим те, которых нет в кэше
            missing = [ts for ts in expected if ts not in existing_set]
            if not missing:
                return []
            # Группируем соседние отсутствующие timestamp в интервалы
            groups = []
            current_group = [missing[0]]
            for ts in missing[1:]:
                if ts - current_group[-1] == interval:
                    current_group.append(ts)
                else:
                    groups.append((current_group[0], current_group[-1] + interval))
                    current_group = [ts]
            groups.append((current_group[0], current_group[-1] + interval))
            return groups

        # Загрузка данных
        for symbol in symbols:
            try:
                session = session_getter(session_cycle)
                if session is None:
                    logger.error(f"Нет доступной сессии для символа {symbol}")
                    continue
                
                # Определяем временной интервал запроса в миллисекундах
                requested_start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
                requested_end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)
                expected_interval = 15 * 60 * 1000  # 15 минут в мс
                
                if symbol in cache and not cache[symbol].empty:
                    logger.info(f"Начало преобразования кэша {symbol} в рабочий DataFrame")
                    # Кэш уже хранит DataFrame, не нужно создавать заново
                    cached_df = cache[symbol]
                    # Проверяем, есть ли уже timestamp_ms
                    if 'timestamp_ms' not in cached_df.columns and 'timestamp' in cached_df.columns:
                        logger.info(f"Векторное преобразование временных меток для {symbol}")
                        # Быстрое векторное преобразование вместо apply
                        cached_df['timestamp_ms'] = pd.to_datetime(cached_df['timestamp'], utc=True).astype('int64') // 1_000_000
                    logger.info(f"Преобразование кэша {symbol} завершено, получено {len(cached_df)} строк")
                    existing_ts = cached_df['timestamp_ms'].tolist()
                    missing_intervals = find_missing_intervals(existing_ts, requested_start_ts, requested_end_ts, expected_interval)
                    new_dfs = []
                    for (miss_start, miss_end) in missing_intervals:
                        start_str = pd.to_datetime(miss_start, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                        end_str = pd.to_datetime(miss_end, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                        logger.info(f"Для {symbol} загружается недостающее окно: {start_str} - {end_str}")
                        new_df = fetch_symbol_data(session=session, symbol=symbol, start_time=start_str, end_time=end_str, chunk_size=chunk_size)
                        if new_df is not None:
                            new_dfs.append(new_df)
                    if new_dfs:
                        combined_df = pd.concat([cached_df.drop(columns=['timestamp_ms'])] + new_dfs, ignore_index=True)
                        combined_df.drop_duplicates(subset=['timestamp', 'symbol'], inplace=True)
                        combined_df.sort_values('timestamp', inplace=True)
                        updated_df = combined_df
                    else:
                        updated_df = cached_df.drop(columns=['timestamp_ms'])
                else:
                    updated_df = fetch_symbol_data(session=session, symbol=symbol, start_time=start_date, end_time=end_date, chunk_size=chunk_size)
                
                if updated_df is not None and not updated_df.empty:
                    processed_count += 1
                    batch_df = save_batch({symbol: updated_df}, processed_count)
                    if batch_df is not None:
                        all_data.append(batch_df)
                
                if processed_count % progress_interval == 0:
                    logger.info(f"Обработано {processed_count}/{len(symbols)} символов")
            except Exception as e:
                logger.error(f"Ошибка при обработке символа {symbol}: {str(e)}")
                continue
        
        # Сохраняем оставшиеся данные
        if _batch_results:
            batch_df = save_batch(_batch_results, (processed_count // batch_size) + 1)
            if batch_df is not None:
                all_data.append(batch_df)
        
        final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        return final_df
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return pd.DataFrame()
        
    finally:
        close_sessions(_sessions)
        cleanup_data(_batch_results, _cached_batch)

def check_disk_space(required_mb: int = 1000) -> bool:
    """Проверка наличия свободного места на диске"""
    try:
        disk = os.path.dirname(os.path.abspath(HISTORICAL_DATA_FILE))
        total, used, free = shutil.disk_usage(disk)
        free_mb = free // (2**20)
        
        if free_mb < required_mb:
            logger.error(f"Недостаточно места на диске: {free_mb}MB < {required_mb}MB")
            return False
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при проверке места на диске: {str(e)}")
        return False

def save_batch(batch_data: Dict[str, pd.DataFrame], batch_number: int) -> Optional[pd.DataFrame]:
    temp_file = None
    backup_file = None
    try:
        # Создаем бэкап текущего файла если он существует
        if os.path.exists(HISTORICAL_DATA_FILE):
            backup_file = f"{HISTORICAL_DATA_FILE}.bak"
            shutil.copy2(HISTORICAL_DATA_FILE, backup_file)
            
        if not batch_data:
            return None
            
        logger.info(f"Сохранение батча {batch_number} ({len(batch_data)} символов)")
        batch_df = pd.concat(batch_data.values(), ignore_index=True)
        
        if not validate_dataframe(batch_df):
            logger.error(f"Некорректный формат данных в батче {batch_number}")
            return None
            
        # Добавить проверку места на диске
        if not check_disk_space():
            logger.error("Недостаточно места на диске для сохранения данных")
            return None
            
        # Используем временный файл для безопасного сохранения
        temp_file = f"{HISTORICAL_DATA_FILE}.temp"
        if os.path.exists(HISTORICAL_DATA_FILE):
            existing_df = pd.read_parquet(HISTORICAL_DATA_FILE)
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            combined_df.to_parquet(temp_file, engine='fastparquet', compression='snappy', index=False)
        else:
            batch_df.to_parquet(temp_file, engine='fastparquet', compression='snappy', index=False)
        
        # Безопасная замена файла
        if os.path.exists(HISTORICAL_DATA_FILE):
            os.replace(temp_file, HISTORICAL_DATA_FILE)
        else:
            os.rename(temp_file, HISTORICAL_DATA_FILE)
            
        logger.info(f"Батч {batch_number} сохранен ({len(batch_df)} записей)")
        
        # Очищаем память после сохранения
        batch_df = batch_df.copy()
        del batch_data
        logger.info(f"Начало очистки памяти (gc.collect) после сохранения батча {batch_number}")
        gc.collect()
        logger.info(f"Очистка памяти после сохранения батча {batch_number} завершена")
        
        return batch_df
        
    except Exception as e:
        # Восстанавливаем из бэкапа при ошибке
        if backup_file and os.path.exists(backup_file):
            shutil.copy2(backup_file, HISTORICAL_DATA_FILE)
        logger.error(f"Ошибка при сохранении батча {batch_number}: {str(e)}")
        return None
        
    finally:
        # Очищаем временные файлы
        for file in [temp_file, backup_file]:
            if file and os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    logger.error(f"Ошибка при удалении временного файла: {str(e)}")
        gc.collect()

def cleanup_data(batch_results: Optional[Dict], cached_batch: Optional[Dict]) -> None:
    """Очистка промежуточных данных"""
    try:
        if isinstance(batch_results, dict):
            batch_results.clear()
        if isinstance(cached_batch, dict):
            cached_batch.clear()
        logger.info("Начало очистки памяти в cleanup_data (gc.collect)")
        gc.collect()
        logger.info("Очистка памяти в cleanup_data завершена")
    except Exception as e:
        logger.error(f"Ошибка при очистке данных: {str(e)}")

def ensure_log_directory(log_file: str) -> None:
    """Создание директории для логов"""
    try:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Ошибка при создании директории логов: {str(e)}")
        raise

def close_sessions(sessions: List[HTTP]) -> None:
    """Закрытие всех сессий"""
    for session in sessions:
        try:
            if hasattr(session, "close") and callable(getattr(session, "close")):
                session.close()
            else:
                logger.info("У объекта HTTP нет метода close, пропускаем закрытие сессии.")
        except Exception as e:
            logger.error(f"Ошибка при закрытии сессии: {str(e)}")

class SafeCache:
    def __init__(self):
        self._cache = {}
        self._lock = Lock()
        
    def get(self, key, default=None):
        with self._lock:
            return self._cache.get(key, default)
            
    def set(self, key, value):
        if value is None:
            return
        with self._lock:
            self._cache[key] = value
            
    def clear(self):
        with self._lock:
            self._cache.clear()

# Использование
safe_cache = SafeCache()

def signal_handler(signum, frame):
    global _sessions, _batch_results, _cached_batch
    logger.info("Получен сигнал завершения, выполняем корректное завершение...")
    try:
        if _sessions:
            close_sessions(_sessions)
        if _batch_results is not None or _cached_batch is not None:
            cleanup_data(_batch_results, _cached_batch)
    except Exception as e:
        logger.error(f"Ошибка при завершении: {str(e)}")
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        ensure_log_directory('data_loader.log')
        
        if not validate_api_keys():
            raise ValueError("Некорректная конфигурация API ключей")
            
        # Загрузка маркетов из файла Markets.txt
        logger.info("Загрузка маркетов из файла Markets.txt...")
        with open('Markets.txt', 'r') as f:
            markets_data = f.read()
            # Разбиваем по запятым и убираем пробелы
            symbols = [s.strip() for s in markets_data.split(',')]
            # Убираем пустые строки
            symbols = [s for s in symbols if s]
            
        logger.info(f"Загружено {len(symbols)} маркетов из Markets.txt")
        
        # Установка дат для загрузки: с 1 января 2022 по 1 июня 2025
        start_date = '2022-01-01'
        end_date = '2025-06-01'
        logger.info(f"Период загрузки данных: с {start_date} по {end_date}")
        
        if symbols:
            logger.info(f"Начинаем загрузку данных для {len(symbols)} символов")
            data = collect_historical_data(symbols, start_date, end_date)
            logger.info(f"Загружено {len(data):,} строк данных")
            logger.info("Загрузка данных успешно завершена")
        else:
            logger.info("Нет символов для загрузки")
            
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        close_sessions(_sessions)
        cleanup_data(_batch_results, _cached_batch)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {str(e)}")
        close_sessions(_sessions)
        cleanup_data(_batch_results, _cached_batch)
