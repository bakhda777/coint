# Стандартные библиотеки
import json, logging, os, configparser, time, random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import ast
import multiprocessing
from functools import partial, lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import cProfile
import io
import pstats

# Make line_profiler optional
try:
    from line_profiler import LineProfiler
except ImportError:
    # We'll log this later after logger is initialized
    line_profiler_missing = True
    # Create a dummy LineProfiler class
    class LineProfiler:
        def __init__(self, *args, **kwargs):
            pass
        def add_function(self, func):
            return self
        def enable_by_count(self):
            pass
        def disable_by_count(self):
            pass
        def print_stats(self, stream=None):
            print("LineProfiler not available")
            
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector, ClientError

# Make pybit optional
try:
    from pybit.unified_trading import HTTP, WebSocket
    from pybit.exceptions import FailedRequestError, InvalidRequestError
except ImportError:
    # We'll log this later after logger is initialized
    pybit_missing = True
    # Create dummy classes for pybit
    class HTTP:
        def __init__(self, *args, **kwargs):
            pass
    class WebSocket:
        def __init__(self, *args, **kwargs):
            pass
    class FailedRequestError(Exception):
        pass
    class InvalidRequestError(Exception):
        pass
from logging import Logger
import sys
import shutil
import psutil
import gc
from logging.handlers import RotatingFileHandler
from threading import Lock
import signal

# Определяем абсолютный путь для файла кэша исторических данных
HISTORICAL_DATA_FILE = os.path.join(os.path.dirname(__file__), 'historical_data.parquet')

# Библиотеки для работы с данными
import pyarrow as pa
import pyarrow.parquet as pq

# Перемещаем в начало файла, после импортов
@lru_cache(maxsize=1000)
def get_rate_limit_delay(current_rps: float) -> float:
    """Оптимизированный расчет задержки для rate limiting"""
    if current_rps > API_RATE_LIMIT:
        return (current_rps - API_RATE_LIMIT) / (API_RATE_LIMIT * 10)
    return 0

# В начале файла, сразу после импортов
def setup_logging():
    """Настройка логирования: файл с JSON форматированием, консоль с human-readable."""
    logger = logging.getLogger()
    
    # Ротация логов
    file_handler = RotatingFileHandler(
        'data_loader.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JsonFormatter())
    
    # Консольный вывод с human-readable форматированием
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'time': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_data)

# Инициализация логгера до его использования
logger = setup_logging()

# Теперь можно использовать logger

# Логируем информацию о недостающих зависимостях
if 'line_profiler_missing' in globals():
    logger.warning("line_profiler not installed. Profiling functionality will be limited.")

if 'pybit_missing' in globals():
    logger.warning("pybit not installed. API functionality will be limited.")

# Сделаем config.ini опциональным
MAINNET_API_KEYS = ['dummy_key1', 'dummy_key2']  # Значения по умолчанию
MAINNET_API_SECRETS = ['dummy_secret1', 'dummy_secret2']  # Значения по умолчанию

if os.path.exists('config.ini'):
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Если файл существует, используем значения из него
        if 'API' in config and 'MAINNET_API_KEY1' in config['API'] and 'MAINNET_API_KEY2' in config['API']:
            MAINNET_API_KEYS = [
                config['API']['MAINNET_API_KEY1'],
                config['API']['MAINNET_API_KEY2']
            ]
        if 'API' in config and 'MAINNET_API_SECRET1' in config['API'] and 'MAINNET_API_SECRET2' in config['API']:
            MAINNET_API_SECRETS = [
                config['API']['MAINNET_API_SECRET1'],
                config['API']['MAINNET_API_SECRET2']
            ]
    except Exception as e:
        logger.warning(f"Ошибка при чтении конфигурации: {str(e)}. Используются значения по умолчанию.")
else:
    logger.warning("Файл конфигурации config.ini не найден. Используются значения по умолчанию.")

def validate_api_keys() -> bool:
    """Проверка корректности API ключей"""
    if not MAINNET_API_KEYS or not MAINNET_API_SECRETS:
        logger.error("API ключи не настроены")
        return False
        
    if len(MAINNET_API_KEYS) != len(MAINNET_API_SECRETS):
        logger.error("Количество API ключей не совпадает с количеством секретов")
        return False
        
    return True

def extract_symbols_from_pairs(pairs_file: str) -> List[Tuple[str, str]]:
    """
    Безопасное извлечение символов из файла пар.
    
    Args:
        pairs_file: Путь к файлу с парами
        
    Returns:
        List[Tuple[str, str]]: Список кортежей с символами пар
        
    Raises:
        ValueError: Если формат файла некорректен
    """
    try:
        with open(pairs_file, 'r') as f:
            content = f.read().strip()
            
        # Удаляем ненужные символы и разбиваем на пары
        pairs_str = content.replace('(', '').replace(')', '').replace(' ', '').replace("'", '').split(',')
        
        # Группируем по парам
        pairs = []
        for i in range(0, len(pairs_str), 2):
            if i + 1 < len(pairs_str):
                pairs.append((pairs_str[i], pairs_str[i + 1]))
        
        # Проверяем формат каждой пары
        validated_pairs = []
        for pair in pairs:
            if len(pair) != 2:
                logger.warning(f"Пропускаем неверный формат пары: {pair}")
                continue
                
            symbol1, symbol2 = pair
            if not symbol1 or not symbol2:
                logger.warning(f"Пропускаем пару с пустыми символами: {pair}")
                continue
                
            validated_pairs.append((symbol1.strip(), symbol2.strip()))
            
        return validated_pairs
        
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла: {str(e)}")

def load_cached_data() -> Dict[str, List]:
    """
    Загружает кэшированные данные из Parquet файла.
    
    Returns:
        Dict[str, List]: Словарь с данными по символам
        
    Raises:
        ValueError: Если структура данных некорректна
    """
    try:
        logger.info(f"Проверяем существование файла кэша по пути: {os.path.abspath(HISTORICAL_DATA_FILE)}")
        if not os.path.exists(HISTORICAL_DATA_FILE):
            logger.info(f"Файл кэша {HISTORICAL_DATA_FILE} не найден")
            return {}
        logger.info(f"Файл кэша найден, размер: {os.path.getsize(HISTORICAL_DATA_FILE)} байт")
            
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        
        # Проверяем обязательные колонки
        required_columns = {'symbol', 'timestamp', 'close', 'volume'}
        if not all(col in df.columns for col in required_columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"В кэше отсутствуют обязательные колонки: {missing}")
            
        # Преобразуем DataFrame в словарь
        cache = {}
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Проверяем непрерывность временного ряда
            timestamps = pd.to_datetime(symbol_data['timestamp'])
            if len(timestamps) > 1:
                time_diff = timestamps.diff().dropna()
                if not (time_diff == time_diff.mode()[0]).all():
                    logger.warning(f"Обнаружены пропуски в данных для {symbol}")
            
            cache[symbol] = symbol_data.to_dict('records')
            
        return cache
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке кэша: {str(e)}")
        return {}

def save_cached_data(cache: Dict[str, List]) -> None:
    """
    Сохраняет данные в Parquet файл.
    
    Args:
        cache: Словарь с данными по символами
    """
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
                    gc.collect()
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
            if symbol not in cache or cache[symbol] != data:
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
        def find_missing_intervals(existing_ts: List[int], start: int, end: int, interval: int) -> List[Tuple[int, int]]:
            # Вычисляем все ожидаемые timestamp с шагом interval
            expected = list(range(start, end + 1, interval))
            existing_set = set(existing_ts)
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
                
                if symbol in cache and cache[symbol]:
                    cached_df = pd.DataFrame(cache[symbol])
                    if 'timestamp' in cached_df.columns:
                        cached_df['timestamp_ms'] = cached_df['timestamp'].apply(lambda x: int(pd.Timestamp(x, tz='UTC').timestamp() * 1000))
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
        gc.collect()
        
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
        gc.collect()
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
            
        # Загрузка пар из файла
        logger.info("Загрузка пар из файла...")
        pairs = extract_symbols_from_pairs('Pairs.txt')
        logger.info(f"Загружено {len(pairs)} пар")
        
        # Извлекаем уникальные символы
        symbols = set()
        for s1, s2 in pairs:
            symbols.add(s1)
            symbols.add(s2)
        symbols = sorted(list(symbols))
        logger.info(f"Всего уникальных символов: {len(symbols)}")
        
        # Проверяем существующие данные
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        
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
