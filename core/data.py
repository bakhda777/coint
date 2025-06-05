"""
Модуль для работы с данными временных рядов.
Содержит класс DataProvider для доступа к данным из Parquet-файлов
и вспомогательные функции для работы с данными.
"""

import logging
from typing import Any, List, Optional, Union, TypeVar, Protocol
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

# Глобальная переменная для доступа к датасету
DATASET = None

# Протокол для определения интерфейса BacktestParams
class BacktestParamsProtocol(Protocol):
    min_token_age_days: int

def _available_symbols(
    window_start_dt: datetime,
    window_end_dt: datetime,
    params: BacktestParamsProtocol, 
    symbols_to_check: list[str] | None = None,  
    min_bars_coverage_days: int = 1,  
    bars_per_day: int = 24 * 4  
) -> list[str]:
    """Возвращает список символов, для которых есть достаточно данных в указанном временном окне
    и которые удовлетворяют критерию минимального возраста.

    Args:
        window_start_dt (datetime): Начало окна.
        window_end_dt (datetime): Конец окна.
        params (BacktestParamsProtocol): Параметры бэктеста, включая min_token_age_days.
        symbols_to_check (list[str] | None, optional): Список символов для проверки.
            Если None, будут проверены все символы из DATASET (может быть медленно).
            Defaults to None.
        min_bars_coverage_days (int): Минимальное количество дней, за которые должны быть данные.
        bars_per_day (int): Ожидаемое количество баров в день.

    Returns:
        list[str]: Список тикеров символов.
    """
    if DATASET is None:
        logging.error("DATASET не инициализирован. Невозможно получить доступные символы.")
        return []

    required_min_bars = min_bars_coverage_days * bars_per_day
    available_symbols_list = []

    # Определяем список символов для проверки
    symbols_for_processing: list[str] | np.ndarray
    if symbols_to_check is not None:
        symbols_for_processing = symbols_to_check
        logging.info(f"Получен предварительно отфильтрованный список из {len(symbols_for_processing)} символов для проверки покрытия.")
    else:
        logging.info("Список symbols_to_check не передан, сканируем все символы из DATASET.")
        try:
            # Это может быть медленно, если символов очень много.
            symbols_for_processing = DATASET.scanner(columns=['symbol']).to_table().column('symbol').to_pandas().unique()
        except Exception as e:
            logging.error(f"Ошибка при получении списка всех символов из DATASET: {e}")
            return []

    if not isinstance(symbols_for_processing, (list, np.ndarray)):
        logging.error(f"Переменная symbols_for_processing не является списком или ndarray: {type(symbols_for_processing)}. Возвращен пустой список.")
        return []
    if len(symbols_for_processing) == 0:
        logging.warning("Список символов для обработки (symbols_for_processing) пуст. Возвращен пустой список.")
        return []

    logging.info(f"Будет проверено {len(symbols_for_processing)} символов на покрытие в окне {window_start_dt.date()} - {window_end_dt.date()}.")

    for symbol_idx, symbol_name in enumerate(symbols_for_processing):
        if (symbol_idx + 1) % 50 == 0: # Логируем прогресс каждые 50 символов
            logging.info(f"Проверено покрытие для {symbol_idx + 1}/{len(symbols_for_processing)} символов...")
        try:
            # Фильтр для выборки данных по символу и дате
            filter_expr = (
                (pc.field("symbol") == pc.scalar(symbol_name)) &
                (pc.field("ts_ms") >= pc.scalar(int(window_start_dt.timestamp() * 1000))) &
                (pc.field("ts_ms") < pc.scalar(int(window_end_dt.timestamp() * 1000)))
            )

            # Считаем количество баров для символа в окне
            table = DATASET.to_table(filter=filter_expr, columns=['ts_ms'])
            num_bars = len(table)

            if num_bars >= required_min_bars:
                # Проверка возраста токена, если параметр задан
                if params.min_token_age_days > 0:
                    try:
                        first_ts_ms_table = DATASET.to_table(
                            filter=(pc.field("symbol") == pc.scalar(symbol_name)),
                            columns=['ts_ms'],
                            limit=1 # Нам нужна только самая первая запись
                        )
                        if len(first_ts_ms_table) > 0:
                            first_ts_ms = first_ts_ms_table['ts_ms'][0].as_py()
                            listing_dt = datetime.fromtimestamp(first_ts_ms / 1000, tz=timezone.utc)
                            token_age_days = (window_start_dt - listing_dt).days

                            if token_age_days >= params.min_token_age_days:
                                available_symbols_list.append(symbol_name)
                        else:
                            pass  # Не удалось определить дату листинга
                    except Exception as age_exc:
                        logging.error(f"Ошибка при проверке возраста токена {symbol_name}: {age_exc}")
                        continue # Пропускаем символ при ошибке определения возраста
                else:
                    # Фильтр по возрасту не применяется
                    available_symbols_list.append(symbol_name)
        except Exception as e:
            logging.error(f"Ошибка при проверке покрытия данных для символа {symbol_name}: {e}")
            continue

    logging.info(f"Найдено {len(available_symbols_list)} символов с достаточным покрытием ({min_bars_coverage_days} дней) в окне {window_start_dt.date()} - {window_end_dt.date()}.")
    return available_symbols_list

class DataProvider:
    """Интерфейс для доступа к данным временных рядов из Parquet-файлов."""
    def __init__(self, data_dir_path: Path | str):
        self.dataset: Optional[ds.Dataset] = None
        self.base_symbols: List[str] = [] # Этот список может заполняться отдельно, если нужно
        self.schema_names: List[str] = []
        self._logger = logging.getLogger(__name__ + ".DataProvider") # Логгер для класса

        if not data_dir_path:
            self._logger.error("Путь к директории с данными не предоставлен для DataProvider.")
            return

        try:
            self.dataset = ds.dataset(str(data_dir_path), format="parquet", partitioning="hive")
            self._logger.info(f"Успешно инициализирован DataProvider из {data_dir_path}. Найдено {len(self.dataset.files)} Parquet файлов.")
            
            if self.dataset:
                self.schema_names = self.dataset.schema.names
                self._logger.debug(f"DataProvider schema names: {self.schema_names}")

        except Exception as e:
            self._logger.error(f"Не удалось инициализировать DataProvider из '{data_dir_path}': {e}")
            self.dataset = None

    def is_ready(self) -> bool:
        """Проверяет, готов ли DataProvider к использованию (датасет загружен)."""
        return self.dataset is not None

    def get_schema_names(self) -> List[str]:
        """Возвращает имена всех колонок в датасете."""
        return self.schema_names if self.dataset else []
        
    def get_files(self) -> List[str]:
        """Возвращает список путей к файлам в датасете."""
        return self.dataset.files if self.dataset else []

    def get_table(self, filter_expression: Optional[pc.Expression] = None, columns: Optional[List[str]] = None) -> Optional[Any]: # pyarrow.Table
        """Извлекает таблицу (или ее часть) из датасета."""
        if not self.is_ready():
            self._logger.error("DataProvider не готов. Невозможно получить таблицу.")
            return None
        try:
            # Приведение self.dataset к типу ds.Dataset для mypy
            current_dataset = self.dataset
            assert current_dataset is not None
            return current_dataset.to_table(filter=filter_expression, columns=columns)
        except Exception as e:
            self._logger.error(f"Ошибка при получении таблицы из DataProvider: {e}")
            return None

    def scan_unique_symbols_from_partitions(self) -> List[str]:
        """Пытается извлечь уникальные имена символов из структуры партиций.
           Это более надежно, чем анализ имен файлов или сканирование колонки 'symbol'.
        """
        if not self.is_ready() or not self.dataset:
            self._logger.warning("DataProvider не готов, невозможно сканировать символы из партиций.")
            return []
        
        try:
            # Проверяем, есть ли 'symbol' в схеме партиционирования
            partitioning_schema = self.dataset.partitioning.schema
            if any(field.name == 'symbol' for field in partitioning_schema):
                # Получаем все выражения партиций
                partition_expressions = self.dataset.partitioning.discover_partitions()
                unique_symbols = set()
                for expr in partition_expressions:
                    # Ищем равенство вида pc.field('symbol') == 'SOME_SYMBOL'
                    if expr.op_name == '==' and expr.args[0].name == 'symbol':
                        unique_symbols.add(expr.args[1].as_py())
                self._logger.info(f"Найдено {len(unique_symbols)} уникальных символов из партиций: {list(unique_symbols)[:10]}...")
                return sorted(list(unique_symbols))
            else:
                self._logger.warning("Колонка 'symbol' не найдена в схеме партиционирования.")
                return []
        except Exception as e:
            self._logger.error(f"Ошибка при сканировании символов из партиций: {e}")
            return []

    def scan_unique_values_from_column(self, column_name: str = 'symbol') -> List[str]:
        """Сканирует и возвращает уникальные значения из указанной колонки датасета."""
        if not self.is_ready():
            self._logger.error(f"DataProvider не готов. Невозможно сканировать колонку '{column_name}'.")
            return []
        
        if column_name not in self.get_schema_names():
            self._logger.error(f"Колонка '{column_name}' не найдена в схеме датасета. Невозможно сканировать.")
            return []
            
        try:
            table = self.dataset.scanner(columns=[column_name]).to_table()
            unique_values_pa = table.column(column_name).unique()
            return [val.as_py() for val in unique_values_pa] # Преобразование в список строк Python
        except Exception as e:
            self._logger.error(f"Ошибка при сканировании колонки '{column_name}' из DataProvider: {e}")
            return []
