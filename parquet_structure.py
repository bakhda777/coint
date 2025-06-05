#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Анализ структуры Parquet-файла

Этот скрипт анализирует структуру Parquet-файла и выводит подробную информацию о его содержимом,
включая названия и типы колонок, наличие пропусков, формат timestamp, распределение тикеров,
статистику по ценам и объёму, а также проверяет временной шаг между записями.
"""

import sys
import pandas as pd
import numpy as np
import os


def inspect_parquet(file_path):
    """
    Функция для анализа структуры Parquet-файла
    
    Args:
        file_path (str): Путь к Parquet-файлу
    """
    print(f"\n{'='*80}")
    print(f"Анализ файла: {file_path}")
    print(f"{'='*80}\n")
    
    # Проверка существования файла
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден!")
        return
    
    try:
        # 1. Чтение Parquet-файла
        print("Чтение Parquet-файла...")
        try:
            df = pd.read_parquet(file_path, engine="pyarrow")
            print("Использован движок: pyarrow")
        except Exception as e:
            print(f"Ошибка при использовании pyarrow: {e}")
            print("Попытка использовать fastparquet...")
            df = pd.read_parquet(file_path, engine="fastparquet")
            print("Использован движок: fastparquet")
        
        # 2. Общая информация о DataFrame
        print("\n--- Общая информация о DataFrame ---")
        print(f"Размер DataFrame: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        # Вывод информации о DataFrame
        print("Количество не-нулевых значений по колонкам:")
        print(df.count())
        print("\nТипы данных и использование памяти:")
        print(df.dtypes)
        
        # 3. Список колонок и их типы
        print("\n--- Названия колонок ---")
        print(df.columns.tolist())
        
        print("\n--- Типы данных ---")
        print(df.dtypes)
        
        # 4. Первые и последние строки
        print("\n--- Первые 5 строк ---")
        print(df.head())
        
        print("\n--- Последние 5 строк ---")
        print(df.tail())
        
        # 5. Проверка пропущенных значений
        print("\n--- Пропущенные значения ---")
        missing = df.isna().sum()
        print(missing)
        
        # 6. Доля пропущенных значений
        print("\n--- Доля пропущенных значений ---")
        missing_percentage = (df.isna().sum() / len(df)).sort_values(ascending=False)
        print(missing_percentage)
        
        # 7. Проверка и преобразование временного столбца
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col:
            print(f"\n--- Обработка временного столбца: {timestamp_col} ---")
            if df[timestamp_col].dtype == 'object' or df[timestamp_col].dtype.name.startswith('datetime') == False:
                print(f"Преобразование {timestamp_col} в datetime...")
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                print(f"Тип после преобразования: {df[timestamp_col].dtype}")
            else:
                print(f"Столбец {timestamp_col} уже имеет тип {df[timestamp_col].dtype}")
            
            # 8. Установка временного столбца в качестве индекса
            print(f"Установка {timestamp_col} в качестве индекса...")
            df_with_time_index = df.set_index(timestamp_col)
            
            # 9. Проверка интервалов между временными метками
            print("\n--- Интервалы между временными метками ---")
            time_diffs = df_with_time_index.index.to_series().diff().dropna()
            print("Топ-5 наиболее частых интервалов:")
            print(time_diffs.value_counts().head(5))
        else:
            print("\nВременной столбец не найден!")
        
        # 10. Описательная статистика
        print("\n--- Описательная статистика ---")
        print(df.describe())
        
        # 11. Проверка наличия колонки symbol
        if 'symbol' in df.columns:
            print("\n--- Анализ колонки 'symbol' ---")
            unique_symbols = df['symbol'].nunique()
            print(f"Уникальных тикеров: {unique_symbols}")
            print("\nТоп-10 тикеров по частоте:")
            print(df['symbol'].value_counts().head(10))
        else:
            # Поиск похожих колонок
            symbol_like_cols = [col for col in df.columns if 'symbol' in col.lower() or 'ticker' in col.lower() or 'asset' in col.lower()]
            if symbol_like_cols:
                print(f"\n--- Анализ колонки '{symbol_like_cols[0]}' ---")
                unique_symbols = df[symbol_like_cols[0]].nunique()
                print(f"Уникальных значений: {unique_symbols}")
                print("\nТоп-10 значений по частоте:")
                print(df[symbol_like_cols[0]].value_counts().head(10))
            else:
                print("\nКолонка с символами/тикерами не найдена!")
        
        # 12. Дополнительная информация о числовых колонках
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print("\n--- Статистика по числовым колонкам ---")
            for col in numeric_cols[:5]:  # Ограничим первыми 5 числовыми колонками для краткости
                print(f"\nКолонка: {col}")
                print(f"Минимум: {df[col].min()}")
                print(f"Максимум: {df[col].max()}")
                print(f"Среднее: {df[col].mean()}")
                print(f"Медиана: {df[col].median()}")
                print(f"Стандартное отклонение: {df[col].std()}")
        
        print(f"\n{'='*80}")
        print("Анализ завершен!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Ошибка при анализе файла: {e}")


if __name__ == "__main__":
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Если путь не указан, используем значение по умолчанию
        file_path = "historical_data_clean.parquet"
    
    inspect_parquet(file_path)
