#!/usr/bin/env python3
"""
Тесты для модуля отчётности.
Проверяем корректность генерации метрик и сохранения отчётов.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reports import (
    trades_to_dataframe,
    calculate_performance_metrics,
    save_backtest_results
)


class TestReports(unittest.TestCase):
    """Тесты для модуля отчётности."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Создаём временную директорию для тестовых файлов
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Очистка после каждого теста."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_trades(self, num_trades=10, initial_capital=10000.0):
        """
        Создаёт образец сделок для тестирования.
        
        Args:
            num_trades: Количество сделок
            initial_capital: Начальный капитал
            
        Returns:
            list: Список словарей с данными сделок
        """
        np.random.seed(42)  # Для воспроизводимости
        
        trades = []
        current_equity = initial_capital
        start_date = datetime(2023, 1, 1)
        
        for i in range(num_trades):
            # Случайные параметры сделки
            entry_date = start_date + timedelta(days=i*2)
            exit_date = entry_date + timedelta(hours=np.random.randint(6, 48))
            
            # PnL: 60% прибыльных сделок
            is_profitable = np.random.random() < 0.6
            if is_profitable:
                pnl = np.random.uniform(10, 100)
            else:
                pnl = -np.random.uniform(5, 80)
            
            # Размеры позиций
            notional_s1 = np.random.uniform(500, 1500)
            notional_s2 = np.random.uniform(500, 1500)
            
            # Обновляем equity
            current_equity += pnl
            
            trade = {
                'position': 'long_spread' if i % 2 == 0 else 'short_spread',
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price_s1': np.random.uniform(40000, 45000),
                'entry_price_s2': np.random.uniform(2500, 3500),
                'exit_price_s1': np.random.uniform(40000, 45000),
                'exit_price_s2': np.random.uniform(2500, 3500),
                'entry_notional_s1': notional_s1,
                'entry_notional_s2': notional_s2,
                'pnl': pnl,
                'pnl_percent': pnl / (notional_s1 + notional_s2),
                'commission': np.random.uniform(2, 10),
                'funding': np.random.uniform(-1, 3),
                'bars_in_trade': int((exit_date - entry_date).total_seconds() / 3600),
                'holding_hours': (exit_date - entry_date).total_seconds() / 3600,
                'exit_reason': np.random.choice(['zscore_exit', 'max_bars', 'stop_loss']),
                'equity_at_entry': current_equity - pnl,
                'equity_at_exit': current_equity,
                'z_score_at_entry': np.random.uniform(-3, 3),
                'z_score_at_exit': np.random.uniform(-1, 1),
                'spread_stats': {
                    'mu_spread': np.random.uniform(-0.1, 0.1),
                    'sigma_spread': np.random.uniform(0.01, 0.1)
                }
            }
            trades.append(trade)
        
        return trades
    
    def test_trades_to_dataframe(self):
        """
        Тест: конвертация списка сделок в DataFrame.
        """
        # Создаём тестовые сделки
        trades = self.create_sample_trades(num_trades=5)
        
        # Конвертируем в DataFrame
        df = trades_to_dataframe(trades)
        
        # Проверяем основные свойства DataFrame
        self.assertIsInstance(df, pd.DataFrame, "Результат должен быть DataFrame")
        self.assertEqual(len(df), 5, "Количество строк должно соответствовать количеству сделок")
        
        # Проверяем обязательные колонки
        required_columns = ['entry_date', 'exit_date', 'pnl', 'position', 'exit_reason']
        for col in required_columns:
            self.assertIn(col, df.columns, f"DataFrame должен содержать колонку {col}")
        
        # Проверяем типы данных
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['entry_date']), 
                       "entry_date должна быть datetime")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['exit_date']), 
                       "exit_date должна быть datetime")
        self.assertTrue(pd.api.types.is_numeric_dtype(df['pnl']), 
                       "pnl должна быть числовой")
        
        # Проверяем, что нет пустых значений в ключевых колонках
        self.assertFalse(df['pnl'].isna().any(), "pnl не должна содержать NaN")
        self.assertFalse(df['entry_date'].isna().any(), "entry_date не должна содержать NaN")
        self.assertFalse(df['exit_date'].isna().any(), "exit_date не должна содержать NaN")
    
    def test_trades_to_dataframe_empty(self):
        """
        Тест: обработка пустого списка сделок.
        """
        empty_trades = []
        df = trades_to_dataframe(empty_trades)
        
        self.assertIsInstance(df, pd.DataFrame, "Результат должен быть DataFrame даже для пустого списка")
        self.assertEqual(len(df), 0, "DataFrame должен быть пустым")
    
    def test_calculate_performance_metrics(self):
        """
        Тест: расчёт метрик производительности.
        """
        # Создаём тестовые сделки
        trades = self.create_sample_trades(num_trades=20, initial_capital=10000.0)
        df = trades_to_dataframe(trades)
        
        # Рассчитываем метрики
        metrics = calculate_performance_metrics(df, initial_capital=10000.0)
        
        # Проверяем основные метрики
        self.assertIsInstance(metrics, dict, "Метрики должны быть в формате словаря")
        
        expected_metrics = [
            'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
            'win_rate', 'profit_factor', 'total_trades',
            'avg_trade_pnl', 'avg_win_pnl', 'avg_loss_pnl'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Метрики должны содержать {metric}")
        
        # Проверяем разумность значений
        self.assertGreaterEqual(metrics['win_rate'], 0.0, "Win rate должен быть >= 0")
        self.assertLessEqual(metrics['win_rate'], 1.0, "Win rate должен быть <= 1")
        
        self.assertGreaterEqual(metrics['max_drawdown_pct'], 0.0, "Max drawdown должен быть >= 0")
        
        self.assertEqual(metrics['total_trades'], 20, "Общее количество сделок должно быть 20")
        
        self.assertGreaterEqual(metrics['profit_factor'], 0.0, "Profit factor должен быть >= 0")
        
        # Проверяем, что Sharpe ratio рассчитан (может быть NaN, если нет вариации)
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float), "Sharpe ratio должен быть числом")
    
    def test_calculate_performance_metrics_all_wins(self):
        """
        Тест: метрики для случая, когда все сделки прибыльные.
        """
        # Создаём только прибыльные сделки
        trades = []
        for i in range(5):
            trade = {
                'entry_date': datetime(2023, 1, 1) + timedelta(days=i),
                'exit_date': datetime(2023, 1, 1) + timedelta(days=i, hours=12),
                'pnl': 50.0 + i * 10,  # Все положительные
                'commission': 2.0,
                'funding': 1.0,
                'holding_hours': 12.0
            }
            trades.append(trade)
        
        df = trades_to_dataframe(trades)
        metrics = calculate_performance_metrics(df, initial_capital=10000.0)
        
        # Проверяем специфичные для этого случая метрики
        self.assertEqual(metrics['win_rate'], 1.0, "Win rate должен быть 100%")
        self.assertEqual(metrics['avg_loss_pnl'], 0.0, "Средний убыток должен быть 0 (нет убыточных сделок)")
        
        # Profit factor должен быть бесконечным (или очень большим) при отсутствии убытков
        self.assertGreater(metrics['profit_factor'], 100, "Profit factor должен быть очень большим")
    
    def test_save_backtest_results(self):
        """
        Тест: сохранение результатов бэктеста в файлы.
        """
        # Создаём тестовые результаты
        trades = self.create_sample_trades(num_trades=10)
        
        backtest_results = {
            'trades': trades,
            'total_pnl': sum(t['pnl'] for t in trades),
            'num_trades': len(trades),
            'execution_time': 5.5,
            'stats': {
                'total_trades': len(trades),
                'win_rate': 0.6,
                'avg_trade_pnl': np.mean([t['pnl'] for t in trades])
            },
            'risk_management': {
                'initial_equity': 10000.0,
                'final_equity': 10000.0 + sum(t['pnl'] for t in trades),
                'max_concurrent_trades': 5
            }
        }
        
        # Сохраняем результаты
        output_dir = Path(self.temp_dir)
        files_created = save_backtest_results(backtest_results, output_dir, "test_backtest")
        
        # Проверяем, что файлы созданы
        self.assertIsInstance(files_created, dict, "Результат должен быть словарём с путями файлов")
        
        expected_file_types = ['parquet', 'csv', 'json']
        for file_type in expected_file_types:
            self.assertIn(file_type, files_created, f"Должен быть создан файл типа {file_type}")
            file_path = files_created[file_type]
            self.assertTrue(os.path.exists(file_path), f"Файл {file_path} должен существовать")
        
        # Проверяем содержимое parquet файла
        parquet_path = files_created['parquet']
        df_loaded = pd.read_parquet(parquet_path)
        self.assertEqual(len(df_loaded), 10, "Загруженный DataFrame должен содержать 10 сделок")
        
        # Проверяем содержимое JSON файла
        import json
        json_path = files_created['json']
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        self.assertIn('summary', json_data, "JSON должен содержать сводку")
        self.assertIn('total_trades', json_data['summary'], "Сводка должна содержать общее количество сделок")
        self.assertEqual(json_data['summary']['total_trades'], 10, "Количество сделок должно быть 10")
    
    def test_performance_metrics_edge_cases(self):
        """
        Тест: обработка крайних случаев в расчёте метрик.
        """
        # Случай 1: Только одна сделка
        single_trade = [{
            'entry_date': datetime(2023, 1, 1),
            'exit_date': datetime(2023, 1, 1, 12),
            'pnl': 50.0,
            'commission': 2.0,
            'funding': 1.0,
            'holding_hours': 12.0
        }]
        
        df_single = trades_to_dataframe(single_trade)
        metrics_single = calculate_performance_metrics(df_single, initial_capital=10000.0)
        
        self.assertEqual(metrics_single['total_trades'], 1, "Должна быть 1 сделка")
        self.assertEqual(metrics_single['win_rate'], 1.0, "Win rate должен быть 100% для прибыльной сделки")
        
        # Случай 2: Все сделки с нулевым PnL
        zero_trades = []
        for i in range(3):
            trade = {
                'entry_date': datetime(2023, 1, 1) + timedelta(days=i),
                'exit_date': datetime(2023, 1, 1) + timedelta(days=i, hours=12),
                'pnl': 0.0,  # Нулевой PnL
                'commission': 2.0,
                'funding': 1.0,
                'holding_hours': 12.0
            }
            zero_trades.append(trade)
        
        df_zero = trades_to_dataframe(zero_trades)
        metrics_zero = calculate_performance_metrics(df_zero, initial_capital=10000.0)
        
        self.assertEqual(metrics_zero['win_rate'], 0.0, "Win rate должен быть 0% для нулевых PnL")
        self.assertEqual(metrics_zero['total_return_pct'], 0.0, "Общая доходность должна быть 0%")


if __name__ == '__main__':
    unittest.main() 