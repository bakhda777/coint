#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов проекта.
Включает тестирование коинтеграции, бэктестинга и отчётности.
"""

import unittest
import sys
import os
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests():
    """
    Обнаруживает и запускает все тесты в директории tests/.
    """
    # Определяем директорию с тестами
    tests_dir = Path(__file__).parent
    
    # Создаём test loader
    loader = unittest.TestLoader()
    
    # Обнаруживаем все тесты
    suite = loader.discover(str(tests_dir), pattern='test_*.py')
    
    # Создаём runner с подробным выводом
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Запускаем тесты
    result = runner.run(suite)
    
    # Возвращаем код выхода на основе результатов
    return 0 if result.wasSuccessful() else 1

def run_specific_test_module(module_name):
    """
    Запускает тесты из конкретного модуля.
    
    Args:
        module_name (str): Имя модуля тестов (например, 'test_cointegration')
    """
    try:
        # Импортируем модуль
        module = __import__(module_name)
        
        # Создаём test suite для этого модуля
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Запускаем тесты
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except ImportError as e:
        print(f"Ошибка импорта модуля {module_name}: {e}")
        return 1

def print_test_summary():
    """Выводит сводку доступных тестов."""
    print("=== ДОСТУПНЫЕ ТЕСТЫ ===")
    print("test_cointegration.py - Тесты модуля коинтеграции")
    print("test_backtest.py      - Smoke-тесты бэктестинга") 
    print("test_reports.py       - Тесты модуля отчётности")
    print()
    print("Использование:")
    print("  python run_tests.py                    # Запустить все тесты")
    print("  python run_tests.py test_cointegration # Запустить конкретный модуль")
    print("  python run_tests.py --summary          # Показать эту справку")

def main():
    """Главная функция."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ['--help', '-h', '--summary']:
            print_test_summary()
            return 0
        else:
            # Запуск конкретного модуля тестов
            return run_specific_test_module(arg)
    else:
        # Запуск всех тестов
        print("=== ЗАПУСК ВСЕХ ТЕСТОВ ===")
        print(f"Директория проекта: {project_root}")
        print(f"Директория тестов: {Path(__file__).parent}")
        print()
        
        exit_code = discover_and_run_tests()
        
        if exit_code == 0:
            print("\n✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        else:
            print("\n❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ!")
        
        return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 