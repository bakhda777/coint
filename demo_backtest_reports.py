#!/usr/bin/env python3
"""
Демонстрационный скрипт для тестирования системы отчётности бэктестинга.
Создаёт синтетические данные сделок и генерирует полные отчёты.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from core.reports import (
    trades_to_dataframe, 
    calculate_performance_metrics,
    create_equity_curve_plot,
    create_trades_analysis_plot,
    generate_html_report,
    save_backtest_results
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_synthetic_trades(num_trades: int = 100, 
                            initial_capital: float = 10000.0,
                            win_rate: float = 0.55) -> list[dict]:
    """
    Генерирует синтетические данные сделок для демонстрации.
    
    Args:
        num_trades: Количество сделок
        initial_capital: Начальный капитал
        win_rate: Процент прибыльных сделок
        
    Returns:
        Список словарей с данными о сделках
    """
    np.random.seed(42)  # Для воспроизводимости
    
    trades = []
    current_equity = initial_capital
    
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_trades):
        # Генерируем случайные даты
        entry_date = start_date + timedelta(days=i*2, hours=np.random.randint(0, 24))
        exit_date = entry_date + timedelta(hours=np.random.randint(1, 48))
        
        # Определяем тип позиции
        position_type = np.random.choice(['long_spread', 'short_spread'])
        
        # Генерируем размеры позиций
        notional_s1 = np.random.uniform(800, 1200)
        notional_s2 = np.random.uniform(800, 1200)
        
        # Генерируем цены входа и выхода
        entry_price_s1 = np.random.uniform(20000, 50000)
        entry_price_s2 = np.random.uniform(1000, 3000)
        
        # Генерируем изменения цен
        price_change_s1 = np.random.normal(0, 0.02)  # 2% волатильность
        price_change_s2 = np.random.normal(0, 0.02)
        
        exit_price_s1 = entry_price_s1 * (1 + price_change_s1)
        exit_price_s2 = entry_price_s2 * (1 + price_change_s2)
        
        # Рассчитываем PnL в зависимости от типа позиции
        if position_type == 'long_spread':
            pnl_s1 = (exit_price_s1 - entry_price_s1) * notional_s1 / entry_price_s1
            pnl_s2 = (entry_price_s2 - exit_price_s2) * notional_s2 / entry_price_s2
        else:  # short_spread
            pnl_s1 = (entry_price_s1 - exit_price_s1) * notional_s1 / entry_price_s1
            pnl_s2 = (exit_price_s2 - entry_price_s2) * notional_s2 / entry_price_s2
        
        total_pnl = pnl_s1 + pnl_s2
        
        # Корректируем PnL в соответствии с заданным win_rate
        if np.random.random() > win_rate:
            # Убыточная сделка
            total_pnl = -abs(total_pnl) * np.random.uniform(0.5, 2.0)
        else:
            # Прибыльная сделка
            total_pnl = abs(total_pnl) * np.random.uniform(0.5, 2.0)
        
        # Рассчитываем комиссии и прочие расходы
        commission = (notional_s1 + notional_s2) * 0.001  # 0.1% комиссия
        funding = np.random.uniform(-10, 20)  # Случайный фандинг
        slippage_cost = (notional_s1 + notional_s2) * 0.0005  # 0.05% проскальзывание
        
        # Финальный PnL с учетом всех расходов
        final_pnl = total_pnl - commission - abs(funding) - slippage_cost
        
        # Обновляем equity
        current_equity += final_pnl
        
        trade = {
            'position': position_type,
            'entry_date': entry_date,
            'entry_price_s1': entry_price_s1,
            'entry_price_s2': entry_price_s2,
            'entry_notional_s1': notional_s1,
            'entry_notional_s2': notional_s2,
            'exit_date': exit_date,
            'exit_price_s1': exit_price_s1,
            'exit_price_s2': exit_price_s2,
            'pnl': final_pnl,
            'pnl_percent': final_pnl / (notional_s1 + notional_s2),
            'commission': commission,
            'funding': funding,
            'slippage': {
                'entry': {'s1_pct': 0.0005, 's2_pct': 0.0005},
                'exit': {'s1_pct': 0.0005, 's2_pct': 0.0005},
                'total_cost': slippage_cost
            },
            'bars_in_trade': np.random.randint(10, 100),
            'holding_hours': (exit_date - entry_date).total_seconds() / 3600,
            'exit_reason': np.random.choice(['zscore_exit', 'max_bars', 'stop_loss', 'end_of_period'], 
                                          p=[0.6, 0.2, 0.1, 0.1]),
            'equity_at_entry': current_equity - final_pnl,
            'equity_at_exit': current_equity,
            'z_score_at_entry': np.random.uniform(-3, 3),
            'z_score_at_exit': np.random.uniform(-1, 1),
            'spread_stats': {
                'mu_spread': np.random.uniform(-0.1, 0.1),
                'sigma_spread': np.random.uniform(0.01, 0.05)
            }
        }
        
        trades.append(trade)
    
    return trades

def demo_reports():
    """Демонстрация системы отчётности."""
    
    logging.info("Начало демонстрации системы отчётности...")
    
    # Генерируем синтетические данные
    logging.info("Генерация синтетических данных сделок...")
    trades_data = generate_synthetic_trades(num_trades=150, win_rate=0.58)
    
    # Создаём результаты бэктестинга в ожидаемом формате
    backtest_results = {
        'trades': trades_data,
        'total_pnl': sum(t['pnl'] for t in trades_data),
        'num_trades': len(trades_data),
        'execution_time': 45.6,
        'risk_management': {
            'initial_equity': 10000.0,
            'final_equity': 10000.0 + sum(t['pnl'] for t in trades_data),
            'max_concurrent_trades': 20,
            'max_capital_usage': 0.8,
            'volatility_sizing': True
        },
        'pair_results': {
            'BTC/ETH': {'num_trades': 45, 'final_pnl_absolute': 234.56},
            'ADA/DOT': {'num_trades': 38, 'final_pnl_absolute': -123.45},
            'SOL/AVAX': {'num_trades': 67, 'final_pnl_absolute': 456.78}
        }
    }
    
    # Конвертируем в DataFrame
    logging.info("Конвертация сделок в DataFrame...")
    trades_df = trades_to_dataframe(trades_data)
    logging.info(f"Создан DataFrame с {len(trades_df)} сделками")
    
    # Рассчитываем метрики производительности
    logging.info("Расчёт метрик производительности...")
    metrics = calculate_performance_metrics(trades_df, initial_capital=10000.0)
    
    # Выводим ключевые метрики
    logging.info("=== РЕЗУЛЬТАТЫ ДЕМОНСТРАЦИИ ===")
    logging.info(f"Общее количество сделок: {metrics['total_trades']}")
    logging.info(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    logging.info(f"Общая доходность: {metrics['total_return_pct']:.2f}%")
    logging.info(f"Коэффициент Шарпа: {metrics['sharpe_ratio']:.2f}")
    logging.info(f"Максимальная просадка: -{metrics['max_drawdown_pct']:.2f}%")
    logging.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logging.info(f"Средний PnL на сделку: ${metrics['avg_trade_pnl']:.2f}")
    logging.info(f"Коэффициент Кальмара: {metrics['calmar_ratio']:.2f}")
    logging.info(f"Коэффициент Сортино: {metrics['sortino_ratio']:.2f}")
    
    # Создаём директорию для результатов
    output_dir = Path("demo_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Генерируем графики
    logging.info("Создание графиков...")
    
    equity_fig = create_equity_curve_plot(
        trades_df, 
        initial_capital=10000.0, 
        save_path=output_dir / "demo_equity_curve.png"
    )
    
    trades_fig = create_trades_analysis_plot(
        trades_df, 
        save_path=output_dir / "demo_trades_analysis.png"
    )
    
    # Закрываем фигуры для экономии памяти
    import matplotlib.pyplot as plt
    plt.close(equity_fig)
    plt.close(trades_fig)
    
    # Сохраняем полные результаты
    logging.info("Сохранение полных результатов...")
    files_created = save_backtest_results(
        backtest_results,
        output_dir,
        "demo_backtest"
    )
    
    logging.info("=== СОЗДАННЫЕ ФАЙЛЫ ===")
    for file_type, file_path in files_created.items():
        logging.info(f"{file_type}: {file_path}")
    
    # Сохраняем также trades в parquet отдельно для демонстрации
    parquet_path = output_dir / "demo_trades_detailed.parquet"
    trades_df.to_parquet(parquet_path)
    logging.info(f"Детальные данные сделок: {parquet_path}")
    
    # Сохраняем также в CSV для удобства просмотра
    csv_path = output_dir / "demo_trades_detailed.csv"
    trades_df.to_csv(csv_path, index=False)
    logging.info(f"Детальные данные сделок (CSV): {csv_path}")
    
    logging.info(f"\nДемонстрация завершена! Откройте файл {files_created['html_report']} в браузере для просмотра отчёта.")
    
    return files_created, metrics

if __name__ == "__main__":
    demo_reports() 