from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def list_partition_symbols(root_data_path: Path) -> list[str]:
    """Сканирует корневую директорию данных и возвращает список имен поддиректорий,
    которые считаются именами символов (например, для hive-партиционированных данных).

    Args:
        root_data_path (Path): Корневой путь к данным (например, 'data/').

    Returns:
        list[str]: Список имен символов (имен поддиректорий).
    """
    if not root_data_path.exists() or not root_data_path.is_dir():
        logger.error(f"Директория данных {root_data_path} не найдена или не является директорией.")
        return []
    
    symbols = []
    for item in root_data_path.iterdir():
        if item.is_dir():
            # Проверяем, содержит ли директория файлы Parquet или другие поддиректории (например, year=...)
            # Это поможет отфильтровать случайные директории, не являющиеся символами.
            # Простая проверка: есть ли вообще что-то внутри, кроме скрытых файлов.
            if any(sub_item for sub_item in item.iterdir() if not sub_item.name.startswith('.')):
                symbols.append(item.name)
            else:
                logger.warning(f"Директория {item.name} в {root_data_path} пуста или содержит только скрытые файлы. Пропускается.")
        
    if not symbols:
        logger.warning(f"Не найдено поддиректорий (символов) в {root_data_path}.")
        
    return symbols
