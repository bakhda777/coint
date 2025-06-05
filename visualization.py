#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для визуализации результатов бэктестинга.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging


def plot_results(trades_df, equity_curve, drawdown):
    """
    Визуализация результатов бэктестинга.
    
    Args:
        trades_df (pandas.DataFrame): Датафрейм с результатами бэктестинга
        equity_curve (pandas.Series): Кривая капитала
        drawdown (pandas.Series): Просадка
    """
    if trades_df.empty:
        logging.info("Нет данных для визуализации")
        return
    
    # Создаем фигуру с несколькими подграфиками
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Кривая капитала
    axes[0].plot(equity_curve.index, equity_curve.values, label='Equity Curve')
    axes[0].set_title('Кривая капитала')
    axes[0].set_ylabel('Капитал (USDT)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Просадка
    axes[1].fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3, label='Drawdown')
    axes[1].set_title('Просадка')
    axes[1].set_ylabel('Просадка (%)')
    axes[1].grid(True)
    axes[1].legend()
    
    # Гистограмма P&L по сделкам
    axes[2].hist(trades_df['pnl'], bins=50, alpha=0.7, label='P&L по сделкам')
    axes[2].axvline(0, color='r', linestyle='--')
    axes[2].set_title('Распределение P&L по сделкам')
    axes[2].set_xlabel('P&L (USDT)')
    axes[2].set_ylabel('Количество сделок')
    axes[2].legend()
    
    # Настройка общих параметров
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    logging.info("График сохранен в backtest_results.png")
    plt.close()


def plot_equity_curve(equity_curve, filename='equity_curve.png'):
    """
    Создание и сохранение графика кривой капитала в PNG формате.
    
    Args:
        equity_curve (pandas.Series): Кривая капитала
        filename (str): Имя файла для сохранения графика
    """
    if equity_curve.empty:
        logging.info("Нет данных для визуализации кривой капитала")
        return
    
    # Создаем график кривой капитала
    plt.figure(figsize=(14, 8))
    
    # Вычисляем начальный и конечный капитал
    initial_capital = equity_curve.iloc[0]
    final_capital = equity_curve.iloc[-1]
    total_return_pct = ((final_capital / initial_capital) - 1) * 100
    
    # График капитала
    plt.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=2)
    
    # Добавляем горизонтальную линию начального капитала
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
    
    # Добавляем аннотации
    plt.text(equity_curve.index[0], initial_capital * 1.05, f'Начальный капитал: {initial_capital:.2f} USDT', 
             color='black', fontsize=10, backgroundcolor='white', alpha=0.7)
    
    # Используем тройные кавычки для многострочной f-строки
    plt.text(equity_curve.index[-1], final_capital * 0.95, 
             f"""Конечный капитал: {final_capital:.2f} USDT
Доходность: {total_return_pct:.2f}%""", 
             color='black', fontsize=10, backgroundcolor='white', alpha=0.7, 
             horizontalalignment='right')
    
    # Настройка графика
    plt.title('Кривая капитала стратегии коинтеграционного трейдинга', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Капитал (USDT)')
    plt.grid(True, alpha=0.3)
    
    # Добавляем форматирование дат
    plt.gcf().autofmt_xdate()
    
    # Сохраняем график
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logging.info(f"График кривой капитала сохранен в {filename}")
    plt.close()


def save_results_to_csv(trades_df, metrics, filename='backtest_results.csv'):
    """
    Сохранение результатов бэктестинга в CSV файл.
    
    Args:
        trades_df (pandas.DataFrame): Датафрейм с результатами бэктестинга
        metrics (dict): Словарь с метриками производительности
        filename (str): Имя файла для сохранения результатов
    """
    if trades_df.empty:
        logging.info("Нет данных для сохранения")
        return
    
    # Сохраняем сделки
    trades_df.to_csv(filename, index=False)
    logging.info(f"Результаты сохранены в {filename}")
    
    # Сохраняем метрики в отдельный файл
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('backtest_metrics.csv', index=False)
    logging.info("Метрики сохранены в backtest_metrics.csv")
