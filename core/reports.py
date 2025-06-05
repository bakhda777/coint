"""
Модуль для генерации отчётов и метрик бэктестинга.
Содержит функции для анализа результатов торговли, построения графиков и создания HTML-отчётов.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def trades_to_dataframe(trades_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Конвертирует список сделок в pandas DataFrame.
    
    Args:
        trades_list: Список словарей с информацией о сделках
        
    Returns:
        pd.DataFrame с данными о сделках
    """
    if not trades_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(trades_list)
    
    # Преобразуем даты в правильный формат
    if 'entry_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
    if 'exit_date' in df.columns:
        df['exit_date'] = pd.to_datetime(df['exit_date'])
    
    # Добавляем полезные колонки
    if 'entry_date' in df.columns and 'exit_date' in df.columns:
        df['holding_days'] = (df['exit_date'] - df['entry_date']).dt.total_seconds() / (24 * 3600)
    
    # Категоризируем сделки
    if 'pnl' in df.columns:
        df['trade_result'] = df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Neutral')
    
    # Добавляем кумулятивный PnL
    if 'pnl' in df.columns:
        df['cumulative_pnl'] = df['pnl'].cumsum()
    
    return df

def calculate_performance_metrics(trades_df: pd.DataFrame, 
                                initial_capital: float = 10000.0,
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Рассчитывает основные метрики производительности торговой стратегии.
    
    Args:
        trades_df: DataFrame с данными о сделках
        initial_capital: Начальный капитал
        risk_free_rate: Безрисковая ставка (годовая)
        
    Returns:
        Словарь с метриками производительности
    """
    metrics = {}
    
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'profit_factor': 0.0,
            'avg_trade_pnl': 0.0,
            'avg_win_pnl': 0.0,
            'avg_loss_pnl': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0
        }
    
    # Базовые метрики
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    metrics['total_trades'] = total_trades
    metrics['win_rate'] = len(wins) / total_trades if total_trades > 0 else 0.0
    metrics['total_pnl'] = trades_df['pnl'].sum()
    metrics['total_return_pct'] = (metrics['total_pnl'] / initial_capital) * 100
    
    # PnL метрики
    metrics['avg_trade_pnl'] = trades_df['pnl'].mean()
    metrics['avg_win_pnl'] = wins['pnl'].mean() if len(wins) > 0 else 0.0
    metrics['avg_loss_pnl'] = losses['pnl'].mean() if len(losses) > 0 else 0.0
    
    # Profit Factor
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.0
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Equity curve для расчёта риск-метрик
    equity_curve = initial_capital + trades_df['cumulative_pnl']
    returns = equity_curve.pct_change().dropna()
    
    # Sharpe Ratio
    if len(returns) > 1 and returns.std() > 0:
        excess_returns = returns - (risk_free_rate / 252)  # Дневная безрисковая ставка
        metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(252)
    else:
        metrics['sharpe_ratio'] = 0.0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_std = downside_returns.std()
        if downside_std > 0:
            excess_returns = returns - (risk_free_rate / 252)
            metrics['sortino_ratio'] = excess_returns.mean() / downside_std * np.sqrt(252)
        else:
            metrics['sortino_ratio'] = float('inf')
    else:
        metrics['sortino_ratio'] = 0.0
    
    # Maximum Drawdown
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max * 100
    metrics['max_drawdown_pct'] = abs(drawdowns.min())
    
    # Calmar Ratio
    if metrics['max_drawdown_pct'] > 0:
        annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
        metrics['calmar_ratio'] = annual_return / (metrics['max_drawdown_pct'] / 100)
    else:
        metrics['calmar_ratio'] = float('inf')
    
    # Consecutive wins/losses
    trade_results = trades_df['pnl'].apply(lambda x: 1 if x > 0 else -1)
    
    def max_consecutive(series, value):
        groups = (series != value).cumsum()
        consecutive_counts = series.groupby(groups).cumsum()
        return consecutive_counts[series == value].max() if (series == value).any() else 0
    
    metrics['max_consecutive_wins'] = max_consecutive(trade_results, 1)
    metrics['max_consecutive_losses'] = max_consecutive(trade_results, -1)
    
    return metrics

def create_equity_curve_plot(trades_df: pd.DataFrame, 
                           initial_capital: float = 10000.0,
                           save_path: Optional[Path] = None) -> Figure:
    """
    Создаёт график equity curve.
    
    Args:
        trades_df: DataFrame с данными о сделках
        initial_capital: Начальный капитал
        save_path: Путь для сохранения графика
        
    Returns:
        matplotlib Figure объект
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    if trades_df.empty:
        ax1.text(0.5, 0.5, 'No trades data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax2.text(0.5, 0.5, 'No drawdown data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        return fig
    
    # Equity curve
    equity_curve = initial_capital + trades_df['cumulative_pnl']
    dates = trades_df['exit_date'] if 'exit_date' in trades_df.columns else range(len(trades_df))
    
    ax1.plot(dates, equity_curve, linewidth=2, color='blue', label='Portfolio Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Форматирование оси X для дат
    if 'exit_date' in trades_df.columns:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Drawdown
    running_max = equity_curve.cummax()
    drawdown_pct = (equity_curve - running_max) / running_max * 100
    
    ax2.fill_between(dates, drawdown_pct, 0, alpha=0.3, color='red')
    ax2.plot(dates, drawdown_pct, color='red', linewidth=1)
    ax2.set_title('Portfolio Drawdown', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    if 'exit_date' in trades_df.columns:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Equity curve saved to {save_path}")
    
    return fig

def create_trades_analysis_plot(trades_df: pd.DataFrame, 
                              save_path: Optional[Path] = None) -> Figure:
    """
    Создаёт анализ сделок (распределение PnL, duration, etc.).
    
    Args:
        trades_df: DataFrame с данными о сделках
        save_path: Путь для сохранения графика
        
    Returns:
        matplotlib Figure объект
    """
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No trades data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PnL Distribution
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] <= 0]['pnl']
    
    if len(wins) > 0:
        ax1.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
    if len(losses) > 0:
        ax1.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.set_title('PnL Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PnL ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Trade Duration
    if 'holding_days' in trades_df.columns:
        ax2.hist(trades_df['holding_days'], bins=20, alpha=0.7, color='blue')
        ax2.set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Holding Period (Days)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
    
    # 3. PnL over Time
    ax3.scatter(range(len(trades_df)), trades_df['pnl'], 
               c=['green' if x > 0 else 'red' for x in trades_df['pnl']], alpha=0.6)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax3.set_title('PnL by Trade Number', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('PnL ($)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Monthly Returns (если есть даты)
    if 'exit_date' in trades_df.columns:
        monthly_pnl = trades_df.set_index('exit_date')['pnl'].resample('M').sum()
        monthly_pnl.plot(kind='bar', ax=ax4, color=['green' if x >= 0 else 'red' for x in monthly_pnl])
        ax4.set_title('Monthly PnL', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('PnL ($)')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Date information not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Trades analysis saved to {save_path}")
    
    return fig

def generate_html_report(backtest_results: Dict[str, Any], 
                        output_dir: Path,
                        report_name: str = "backtest_report") -> Path:
    """
    Генерирует HTML отчёт с результатами бэктестинга.
    
    Args:
        backtest_results: Результаты бэктестинга
        output_dir: Директория для сохранения отчёта
        report_name: Имя файла отчёта (без расширения)
        
    Returns:
        Path к созданному HTML файлу
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Конвертируем сделки в DataFrame
    trades_df = trades_to_dataframe(backtest_results.get('trades', []))
    
    # Рассчитываем метрики
    initial_capital = backtest_results.get('risk_management', {}).get('initial_equity', 10000.0)
    metrics = calculate_performance_metrics(trades_df, initial_capital)
    
    # Создаём графики
    equity_fig = create_equity_curve_plot(trades_df, initial_capital, 
                                        output_dir / f"{report_name}_equity.png")
    trades_fig = create_trades_analysis_plot(trades_df, 
                                           output_dir / f"{report_name}_trades.png")
    
    plt.close(equity_fig)
    plt.close(trades_fig)
    
    # Сохраняем DataFrame в parquet
    if not trades_df.empty:
        parquet_path = output_dir / f"{report_name}_trades.parquet"
        trades_df.to_parquet(parquet_path)
        logging.info(f"Trades data saved to {parquet_path}")
    
    # Генерируем HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report - {report_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
            .metric-label {{ color: #666; margin-top: 5px; }}
            .section {{ margin-bottom: 40px; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            .summary-table {{ width: 100%; border-collapse: collapse; }}
            .summary-table th, .summary-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            .summary-table th {{ background-color: #f2f2f2; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Cointegration Strategy Backtest Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Key Performance Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value {'positive' if metrics['total_return_pct'] >= 0 else 'negative'}">
                        {metrics['total_return_pct']:.2f}%
                    </div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">
                        {metrics['sharpe_ratio']:.2f}
                    </div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value negative">
                        -{metrics['max_drawdown_pct']:.2f}%
                    </div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">
                        {metrics['win_rate']*100:.1f}%
                    </div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">
                        {metrics['total_trades']}
                    </div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">
                        {metrics['profit_factor']:.2f}
                    </div>
                    <div class="metric-label">Profit Factor</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Equity Curve & Drawdown</h2>
            <div class="chart">
                <img src="{report_name}_equity.png" alt="Equity Curve">
            </div>
        </div>
        
        <div class="section">
            <h2>Trade Analysis</h2>
            <div class="chart">
                <img src="{report_name}_trades.png" alt="Trade Analysis">
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Metrics</h2>
            <table class="summary-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{metrics['total_trades']}</td></tr>
                <tr><td>Win Rate</td><td>{metrics['win_rate']*100:.2f}%</td></tr>
                <tr><td>Total PnL</td><td class="{'positive' if metrics['total_pnl'] >= 0 else 'negative'}">${metrics['total_pnl']:.2f}</td></tr>
                <tr><td>Total Return</td><td class="{'positive' if metrics['total_return_pct'] >= 0 else 'negative'}">{metrics['total_return_pct']:.2f}%</td></tr>
                <tr><td>Average Trade PnL</td><td class="{'positive' if metrics['avg_trade_pnl'] >= 0 else 'negative'}">${metrics['avg_trade_pnl']:.2f}</td></tr>
                <tr><td>Average Win</td><td class="positive">${metrics['avg_win_pnl']:.2f}</td></tr>
                <tr><td>Average Loss</td><td class="negative">${metrics['avg_loss_pnl']:.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{metrics['profit_factor']:.2f}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{metrics['sharpe_ratio']:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{metrics['sortino_ratio']:.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{metrics['calmar_ratio']:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">-{metrics['max_drawdown_pct']:.2f}%</td></tr>
                <tr><td>Max Consecutive Wins</td><td>{metrics['max_consecutive_wins']}</td></tr>
                <tr><td>Max Consecutive Losses</td><td>{metrics['max_consecutive_losses']}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Strategy Parameters</h2>
            <table class="summary-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Initial Capital</td><td>${initial_capital:.2f}</td></tr>
                <tr><td>Max Concurrent Trades</td><td>{backtest_results.get('risk_management', {}).get('max_concurrent_trades', 'N/A')}</td></tr>
                <tr><td>Max Capital Usage</td><td>{backtest_results.get('risk_management', {}).get('max_capital_usage', 'N/A')}</td></tr>
                <tr><td>Volatility Sizing</td><td>{backtest_results.get('risk_management', {}).get('volatility_sizing', 'N/A')}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Files Generated</h2>
            <ul>
                <li><strong>{report_name}_trades.parquet</strong> - Raw trades data</li>
                <li><strong>{report_name}_equity.png</strong> - Equity curve chart</li>
                <li><strong>{report_name}_trades.png</strong> - Trade analysis charts</li>
                <li><strong>{report_name}.html</strong> - This report</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Сохраняем HTML файл
    html_path = output_dir / f"{report_name}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"HTML report generated: {html_path}")
    return html_path

def save_backtest_results(backtest_results: Dict[str, Any], 
                         output_dir: Path,
                         filename_prefix: str = "backtest") -> Dict[str, Path]:
    """
    Сохраняет результаты бэктестинга в различных форматах.
    
    Args:
        backtest_results: Результаты бэктестинга
        output_dir: Директория для сохранения
        filename_prefix: Префикс имён файлов
        
    Returns:
        Словарь с путями к созданным файлам
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_created = {}
    
    # Конвертируем сделки в DataFrame и сохраняем
    trades_df = trades_to_dataframe(backtest_results.get('trades', []))
    
    if not trades_df.empty:
        # Parquet для эффективного хранения
        parquet_path = output_dir / f"{filename_prefix}_trades.parquet"
        trades_df.to_parquet(parquet_path)
        files_created['trades_parquet'] = parquet_path
        
        # CSV для удобства просмотра
        csv_path = output_dir / f"{filename_prefix}_trades.csv"
        trades_df.to_csv(csv_path, index=False)
        files_created['trades_csv'] = csv_path
    
    # Сохраняем полные результаты в JSON
    import json
    
    # Подготавливаем данные для JSON (удаляем несериализуемые объекты)
    json_data = {
        'total_pnl': backtest_results.get('total_pnl', 0.0),
        'num_trades': backtest_results.get('num_trades', 0),
        'execution_time': backtest_results.get('execution_time', 0.0),
        'risk_management': backtest_results.get('risk_management', {}),
        'stats': backtest_results.get('stats', {}),
        'pair_results': {}
    }
    
    # Добавляем результаты по парам (без детальных trades)
    for pair_name, pair_result in backtest_results.get('pair_results', {}).items():
        json_data['pair_results'][pair_name] = {
            'num_trades': pair_result.get('num_trades', 0),
            'final_pnl_pct': pair_result.get('final_pnl_pct', 0.0),
            'final_pnl_absolute': pair_result.get('final_pnl_absolute', 0.0),
            'formation_metrics': pair_result.get('formation_metrics'),
            'error_message': pair_result.get('error_message')
        }
    
    json_path = output_dir / f"{filename_prefix}_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    files_created['summary_json'] = json_path
    
    # Генерируем HTML отчёт
    html_path = generate_html_report(backtest_results, output_dir, filename_prefix)
    files_created['html_report'] = html_path
    
    logging.info(f"Backtest results saved to {output_dir}")
    return files_created 