#!/usr/bin/env python3
"""
Тесты для модуля коинтеграции.
Проверяем корректность расчётов τ-статистики и обнаружения коинтеграции.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Добавляем путь к модулю
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import compute_coint_and_beta, BacktestParams


class TestCointegration(unittest.TestCase):
    """Тесты для функций коинтеграции."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        self.params = BacktestParams()
        self.params.min_data_points_for_coint = 30
        self.params.min_data_points_for_adf_test = 20
        self.params.adf_max_lag = 5
        
        # Создаём временную директорию для данных
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Очистка после каждого теста."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_synthetic_cointegrated_data(self, n_points=100, beta=1.5, noise_level=0.1):
        """
        Создаёт синтетические коинтегрированные данные.
        
        Args:
            n_points: Количество точек данных
            beta: Коэффициент коинтеграции 
            noise_level: Уровень шума
            
        Returns:
            tuple: (log_prices_s1, log_prices_s2, timestamps)
        """
        # Генерируем случайное блуждание для первого ряда
        np.random.seed(42)  # Для воспроизводимости
        
        # Базовый тренд
        t = np.arange(n_points)
        trend = 0.001 * t
        
        # Случайные шоки
        shocks = np.random.normal(0, 0.02, n_points)
        
        # Первый ряд: случайное блуждание с трендом
        log_prices_s1 = np.cumsum(shocks) + trend + 4.0  # Начинаем с логарифма ~54
        
        # Второй ряд: коинтегрирован с первым + стационарный шум
        stationary_noise = np.random.normal(0, noise_level, n_points)
        # Добавляем AR(1) компонент к шуму для более реалистичной структуры
        for i in range(1, len(stationary_noise)):
            stationary_noise[i] += 0.3 * stationary_noise[i-1]
        
        log_prices_s2 = (log_prices_s1 - 0.5) / beta + stationary_noise
        
        # Создаём временные метки
        start_time = datetime(2023, 1, 1)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
        
        return log_prices_s1, log_prices_s2, timestamps
    
    def create_synthetic_non_cointegrated_data(self, n_points=100):
        """
        Создаёт синтетические НЕ коинтегрированные данные (два независимых случайных блуждания).
        """
        np.random.seed(123)  # Другой seed для независимости
        
        # Два независимых случайных блуждания
        shocks1 = np.random.normal(0, 0.02, n_points)
        shocks2 = np.random.normal(0, 0.025, n_points)
        
        log_prices_s1 = np.cumsum(shocks1) + 4.0
        log_prices_s2 = np.cumsum(shocks2) + 3.5
        
        start_time = datetime(2023, 1, 1)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
        
        return log_prices_s1, log_prices_s2, timestamps
    
    def create_mock_data_files(self, log_prices_s1, log_prices_s2, timestamps, symbol1="BTCUSDT", symbol2="ETHUSDT"):
        """
        Создаёт mock файлы данных в формате, ожидаемом системой.
        
        Returns:
            tuple: (symbol1, symbol2) для использования в тестах
        """
        # Создаём DataFrame для каждого символа
        df1 = pd.DataFrame({
            'timestamp': timestamps,
            'close': np.exp(log_prices_s1),  # Конвертируем обратно в обычные цены
            'volume_usdt': np.random.uniform(1000000, 5000000, len(timestamps))
        })
        
        df2 = pd.DataFrame({
            'timestamp': timestamps, 
            'close': np.exp(log_prices_s2),
            'volume_usdt': np.random.uniform(1000000, 5000000, len(timestamps))
        })
        
        # Сохраняем в parquet файлы (имитируем структуру реальных данных)
        symbol1_dir = os.path.join(self.temp_dir, symbol1)
        symbol2_dir = os.path.join(self.temp_dir, symbol2)
        
        os.makedirs(symbol1_dir, exist_ok=True)
        os.makedirs(symbol2_dir, exist_ok=True)
        
        df1.to_parquet(os.path.join(symbol1_dir, "2023-01.parquet"))
        df2.to_parquet(os.path.join(symbol2_dir, "2023-01.parquet"))
        
        return symbol1, symbol2
    
    def test_cointegrated_pair_detection(self):
        """
        Тест: заведомо коинтегрированная пара должна иметь τ < критического значения.
        """
        # Создаём коинтегрированные данные
        log_s1, log_s2, timestamps = self.create_synthetic_cointegrated_data(
            n_points=100, beta=1.5, noise_level=0.05
        )
        
        # Мокаем функцию получения данных
        import unittest.mock as mock
        
        with mock.patch('prices.get_log_prices') as mock_get_prices:
            # Возвращаем наши синтетические данные
            log_prices_array = np.array([log_s1, log_s2])
            mock_get_prices.return_value = (log_prices_array, pd.DatetimeIndex(timestamps))
            
            # Запускаем тест коинтеграции
            result = compute_coint_and_beta(
                pair=("MOCK1", "MOCK2"),
                start_dt=datetime(2023, 1, 1),
                end_dt=datetime(2023, 1, 10),
                params=self.params
            )
        
        # Проверяем результаты
        self.assertIsNotNone(result, "Результат не должен быть None")
        self.assertIsNotNone(result['tau_stat'], "τ-статистика должна быть рассчитана")
        self.assertIsNotNone(result['beta'], "Коэффициент β должен быть рассчитан")
        
        # Для коинтегрированной пары τ должна быть отрицательной и значительной
        self.assertLess(result['tau_stat'], -1.5, 
                       f"Для коинтегрированной пары τ должна быть < -1.5, получено: {result['tau_stat']}")
        
        # Проверяем, что β близко к истинному значению (1.5)
        self.assertAlmostEqual(result['beta'], 1.5, delta=0.3,
                              f"β должна быть близко к 1.5, получено: {result['beta']}")
        
        # Проверяем флаг коинтеграции
        self.assertTrue(result['is_coint'], "Пара должна быть определена как коинтегрированная")
        
        # Проверяем статистики спреда
        self.assertIsNotNone(result['mu_spread'], "Среднее спреда должно быть рассчитано")
        self.assertIsNotNone(result['sigma_spread'], "СКО спреда должно быть рассчитано")
        self.assertGreater(result['sigma_spread'], 0, "СКО спреда должно быть > 0")
    
    def test_non_cointegrated_pair_detection(self):
        """
        Тест: НЕ коинтегрированная пара должна иметь τ близкую к нулю или положительную.
        """
        # Создаём НЕ коинтегрированные данные
        log_s1, log_s2, timestamps = self.create_synthetic_non_cointegrated_data(n_points=100)
        
        import unittest.mock as mock
        
        with mock.patch('prices.get_log_prices') as mock_get_prices:
            log_prices_array = np.array([log_s1, log_s2])
            mock_get_prices.return_value = (log_prices_array, pd.DatetimeIndex(timestamps))
            
            result = compute_coint_and_beta(
                pair=("MOCK1", "MOCK2"),
                start_dt=datetime(2023, 1, 1),
                end_dt=datetime(2023, 1, 10),
                params=self.params
            )
        
        # Проверяем результаты
        self.assertIsNotNone(result, "Результат не должен быть None")
        self.assertIsNotNone(result['tau_stat'], "τ-статистика должна быть рассчитана")
        
        # Для НЕ коинтегрированной пары τ должна быть > -2.5 (менее значительная)
        self.assertGreater(result['tau_stat'], -2.5,
                          f"Для НЕ коинтегрированной пары τ должна быть > -2.5, получено: {result['tau_stat']}")
        
        # Проверяем флаг коинтеграции (должен быть False)
        self.assertFalse(result['is_coint'], "Пара НЕ должна быть определена как коинтегрированная")
    
    def test_insufficient_data_handling(self):
        """
        Тест: корректная обработка случая с недостаточным количеством данных.
        """
        # Создаём очень мало данных (меньше минимума)
        log_s1, log_s2, timestamps = self.create_synthetic_cointegrated_data(n_points=10)
        
        import unittest.mock as mock
        
        with mock.patch('prices.get_log_prices') as mock_get_prices:
            log_prices_array = np.array([log_s1, log_s2])
            mock_get_prices.return_value = (log_prices_array, pd.DatetimeIndex(timestamps))
            
            result = compute_coint_and_beta(
                pair=("MOCK1", "MOCK2"),
                start_dt=datetime(2023, 1, 1),
                end_dt=datetime(2023, 1, 10),
                params=self.params
            )
        
        # Проверяем, что функция корректно обрабатывает недостаток данных
        self.assertIsNotNone(result, "Результат не должен быть None даже при недостатке данных")
        self.assertEqual(result['n_obs'], 10, "Количество наблюдений должно быть 10")
        self.assertIsNotNone(result['coint_error_msg'], "Должно быть сообщение об ошибке")
        self.assertIn("Insufficient data", result['coint_error_msg'])
    
    def test_no_data_handling(self):
        """
        Тест: корректная обработка случая с отсутствием данных.
        """
        import unittest.mock as mock
        
        with mock.patch('prices.get_log_prices') as mock_get_prices:
            # Имитируем отсутствие данных
            mock_get_prices.return_value = None
            
            result = compute_coint_and_beta(
                pair=("NONEXISTENT1", "NONEXISTENT2"),
                start_dt=datetime(2023, 1, 1),
                end_dt=datetime(2023, 1, 10),
                params=self.params
            )
        
        # Проверяем корректную обработку отсутствия данных
        self.assertIsNotNone(result, "Результат не должен быть None")
        self.assertEqual(result['n_obs'], 0, "Количество наблюдений должно быть 0")
        self.assertIsNotNone(result['coint_error_msg'], "Должно быть сообщение об ошибке")
        self.assertIn("No log prices", result['coint_error_msg'])


if __name__ == '__main__':
    unittest.main() 