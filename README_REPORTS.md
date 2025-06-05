# Система отчётности для бэктестинга pairs trading

## Возможности

✅ **Автоматический сбор и сохранение данных сделок:**
- Конвертация в pandas DataFrame
- Сохранение в parquet и CSV форматах
- JSON сводка результатов

✅ **Расчёт ключевых метрик производительности:**
- Общая доходность (Total Return)
- Коэффициент Шарпа (Sharpe Ratio)
- Максимальная просадка (Maximum Drawdown)
- Hit Rate (процент прибыльных сделок)
- Profit Factor 
- Коэффициенты Кальмара и Сортино
- Статистики по времени удержания позиций

✅ **Визуализация результатов:**
- Equity curve (кривая капитала)
- Распределение PnL сделок
- Месячная доходность
- Анализ drawdown периодов

✅ **HTML отчёт:**
- Интерактивный отчёт с графиками
- Таблицы с ключевыми метриками
- Готов для просмотра в браузере

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Импорт модуля отчётности

```python
from core.reports import (
    trades_to_dataframe,
    calculate_performance_metrics,
    save_backtest_results
)
```

### 2. Использование интегрированной функции

```python
from backtest import run_backtest_with_reports, BacktestParams
from datetime import datetime

# Параметры бэктестинга
params = BacktestParams()

# Пары для тестирования
pairs = [
    ("BTCUSDT", "ETHUSDT"),
    ("ADAUSDT", "DOTUSDT"),
]

# Периоды
formation_start = datetime(2023, 1, 1)
formation_end = datetime(2023, 3, 31)
trading_start = datetime(2023, 4, 1)
trading_end = datetime(2023, 6, 30)

# Запуск с автоматической генерацией отчётов
results = run_backtest_with_reports(
    pairs=pairs,
    formation_start_dt=formation_start,
    formation_end_dt=formation_end,
    trading_start_dt=trading_start,
    trading_end_dt=trading_end,
    params=params,
    output_dir="backtest_results",
    report_name="my_backtest"
)

print(f"HTML отчёт: {results['output_files']['html_report']}")
```

### 3. Ручное создание отчётов

```python
# Если у вас уже есть список сделок
trades_df = trades_to_dataframe(your_trades_list)

# Расчёт метрик
metrics = calculate_performance_metrics(trades_df, initial_capital=10000)

# Сохранение результатов
backtest_results = {
    'trades': your_trades_list,
    'total_pnl': sum(t['pnl'] for t in your_trades_list),
    'num_trades': len(your_trades_list),
    # ... другие поля
}

files = save_backtest_results(
    backtest_results, 
    output_dir="results", 
    report_name="test"
)
```

## Демонстрация

Запустите демо с синтетическими данными:

```bash
python demo_backtest_reports.py
```

Это создаст папку `demo_reports/` с примерами всех типов отчётов.

## Структура выходных файлов

После запуска бэктестинга будут созданы следующие файлы:

```
backtest_results/
├── my_backtest.html              # HTML отчёт (главный файл)
├── my_backtest_trades.parquet    # Данные сделок (parquet)
├── my_backtest_trades.csv        # Данные сделок (CSV)
├── my_backtest_summary.json      # JSON сводка
├── my_backtest_equity.png        # График equity curve
└── my_backtest_trades.png        # График анализа сделок
```

## Ключевые метрики

Система автоматически рассчитывает:

- **Total Return** - общая доходность в %
- **Sharpe Ratio** - отношение доходности к волатильности
- **Max Drawdown** - максимальная просадка в %
- **Win Rate** - процент прибыльных сделок
- **Profit Factor** - отношение прибыли к убыткам
- **Calmar Ratio** - доходность/макс.просадка
- **Sortino Ratio** - модификация Sharpe только для негативной волатильности
- **Average Trade PnL** - средний результат сделки
- **Max/Min Trade PnL** - лучшая/худшая сделка
- **Average Holding Time** - среднее время удержания позиций

## Настройки

Параметры в `run_backtest_with_reports()`:

- `save_trades_parquet` - сохранять ли parquet файлы
- `generate_charts` - создавать ли графики  
- `generate_html_report` - генерировать ли HTML отчёт
- `output_dir` - директория для сохранения
- `report_name` - префикс имён файлов

## Примечания

- HTML отчёт содержит встроенные изображения (base64), поэтому его можно передавать как один файл
- Parquet формат рекомендуется для больших объёмов данных
- CSV файлы удобны для анализа в Excel/Google Sheets
- Все времена в UTC
- Поддерживаются как долгие, так и короткие спред-позиции 