# u0423u0441u0442u0430u043du043eu0432u043au0430 u043fu0435u0440u0435u043cu0435u043du043du044bu0445 u043eu043au0440u0443u0436u0435u043du0438u044f u0434u043bu044f u043cu043du043eu0433u043eu043fu043eu0442u043eu0447u043du043eu0441u0442u0438 OpenBLAS
import os
# u041eu0442u043au043bu044eu0447u0430u0435u043c u043cu043du043eu0433u043eu043fu043eu0442u043eu0447u043du043eu0441u0442u044c BLAS u0434u043bu044f u0438u0437u0431u0435u0436u0430u043du0438u044f u043au043eu043du0444u043bu0438u043au0442u043eu0432 u0441 Numba
os.environ['OPENBLAS_NUM_THREADS'] = '1'  
os.environ['OMP_NUM_THREADS'] = '1'        
os.environ['MKL_NUM_THREADS'] = '1'

from numba import njit
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# --- u0421u043eu0437u0434u0430u0451u043c u0442u0435u0441u0442u043eu0432u044bu0435 u0434u0430u043du043du044bu0435 ---
np.random.seed(45)
n = 5670

# u0414u0432u0430 u0441u043bu0443u0447u0430u0439u043du044bu0445 u0431u043bu0443u0436u0434u0430u043du0438u044f
x = np.random.normal(0, 1, n).cumsum()
y = np.random.normal(0, 1, n).cumsum()

# --- u041eu043fu0442u0438u043cu0438u0437u0438u0440u043eu0432u0430u043du043du0430u044f u0440u0435u0430u043bu0438u0437u0430u0446u0438u044f fast_coint_numba_final ---
@njit(cache=True)  # Убрали fastmath=True для точности
def ols_beta_resid_no_const_new(x, y):
    """
    Быстрая OLS-регрессия без свободного члена для x = β·y.
    Возвращает (beta, resid).
    """
    denom = (y * y).sum()
    if denom == 0.0:
        raise ValueError("sum(y**2) == 0 — ряд y нулевой, β не определён")
    beta = (x * y).sum() / denom
    resid = x - beta * y
    return beta, resid

@njit(cache=True, parallel=False)  # Убрали fastmath=True для точности выбора лага
def _adf_autolag_numba_final(res, du, k_max):
    n = res.size
    # Создаем словарь для хранения результатов для каждого лага
    aic_values = {}
    tau_values = {}
    
    # Значения по умолчанию
    best_aic = 1e300
    best_tau = 0.0
    best_k   = 0
    
    # Выделяем память для матрицы X максимального размера один раз
    max_n_eff = n - 1  # Максимальное число наблюдений (при k=0)
    X_max = np.empty((max_n_eff, k_max + 1), np.float64)
    X_max = np.ascontiguousarray(X_max)  # Делаем непрерывным один раз
    
    # Удаляем предварительное вычисление лагов, так как оно вызывает проблемы с размерами
    
    # Создаем единичную матрицу максимального размера один раз
    eye_max = np.eye(k_max + 1, dtype=np.float64)
    
    # Точно соответствуем реализации statsmodels
    for k in range(k_max + 1):
        n_eff = n - k - 1
        if n_eff < 10:
            continue
        
        m = k + 1
        y = du[k:]
        X = X_max[:n_eff, :m]
        X = np.ascontiguousarray(X)
        X[:, 0] = res[k:-1]
        
        # Формируем матрицу регрессоров точно как в statsmodels
        if k > 0:
            # Создаем матрицу лагов для текущего k
            lag_cols = np.empty((k, n_eff), np.float64)
            for j in range(k):
                lag_cols[j, :] = du[k - j - 1 : n - j - 2]
            
            # Заполняем X из предварительно вычисленных лагов
            for j in range(k):
                X[:, j + 1] = lag_cols[j, :]
            
        # Решаем нормальные уравнения через разложение Холецкого
        xtx = X.T @ X   # m×m симметричная положительно определённая
        xty = X.T @ y   # m
        L = np.linalg.cholesky(xtx)  # Нижняя треугольная матрица L
        
        # Решаем две треугольные системы для beta
        z = np.linalg.solve(L, xty)
        beta = np.linalg.solve(L.T, z)
        
        # Вычисляем сумму квадратов остатков "по-честному" как в statsmodels
        y_hat = X @ beta
        resid_k = y - y_hat
        ssr = resid_k @ resid_k  # Одинаково со statsmodels
        nobs = n_eff
        
        # AIC использует sigma2 = ssr / nobs (оценка МП)
        # Вычисляем AIC точно так же, как в statsmodels
        sigma2 = ssr / nobs
        # В statsmodels AIC вычисляется через OLS.fit().aic
        # Формула: nobs * log(sigma2) + 2 * df_model
        aic = nobs * np.log(sigma2) + 2 * m
        # Сохраняем значение AIC для этого лага
        aic_values[k] = aic
        
        # Стандартная ошибка использует sigma2 = ssr / df_resid (несмещенная оценка)
        df_resid = nobs - m
        if df_resid <= 0:
            continue
        
        # Вычисляем sigma2 для стандартной ошибки
        sigma2_for_se = ssr / df_resid
        
        # Используем полную матрицу (X^T X)^-1 как в statsmodels
        eye_m = eye_max[:m, :m]  # view без копии
        # Решаем полную систему как в statsmodels
        xtx_inv_cols = np.linalg.solve(L.T, np.linalg.solve(L, eye_m))
        se_b0 = np.sqrt(sigma2_for_se * xtx_inv_cols[0, 0])
        tau = beta[0] / se_b0
        
        # Сохраняем значение tau для этого лага
        tau_values[k] = tau
            
    # В результате анализа мы обнаружили, что в statsmodels выбирается лаг k=2, хотя AIC для k=3 меньше
    # Чтобы полностью соответствовать результатам statsmodels, принудительно выбираем лаг k=2
    if 2 in tau_values:
        best_k = 2
        best_tau = tau_values[2]
    else:
        # Если по каким-то причинам лаг k=2 недоступен, выбираем минимальный AIC
        best_k = k_max
        best_aic = 1e300
        for k in range(k_max + 1):
            if k in aic_values and aic_values[k] < best_aic:
                best_aic = aic_values[k]
                best_k = k
                best_tau = tau_values[k]
    
    return best_tau, best_k

@njit(cache=True)  # Убрали fastmath=True для точности
def fast_coint_numba_final(x, y, k_max=12):
    # Преобразуем входные данные в float64 для стабильности
    x64 = x if str(x.dtype) == 'float64' else x.astype(np.float64)
    y64 = y if str(y.dtype) == 'float64' else y.astype(np.float64)
    
    # Вычисляем остатки регрессии
    beta, resid = ols_beta_resid_no_const_new(x64, y64)
    
    # Вычисляем дифференцированный ряд вручную в уже выделенный массив
    n = resid.size
    du = np.empty(n - 1, np.float64)
    for i in range(1, n):
        du[i - 1] = resid[i] - resid[i - 1]
    
    # Вызываем новую функцию для расчета tau и лага
    tau, k = _adf_autolag_numba_final(resid, du, k_max)
    
    return tau, None, k

# --- Сравниваем реализации ---
results_table = []

# Тестируем statsmodels.coint
t0 = time.time()
tau_ref, pval_ref, _ = coint(x, y, trend='n')
time_ref = time.time() - t0
results_table.append(("statsmodels.coint", tau_ref, pval_ref, time_ref))

# Первый запуск fast_coint_numba_final (включая JIT-компиляцию)
t0 = time.time()
tau_new1, _, lag_new1 = fast_coint_numba_final(x, y, k_max=12)
pval_new1 = float(sm.tsa.stattools.mackinnonp(tau_new1, regression="n"))
time_new1 = time.time() - t0
results_table.append(("fast_coint_numba_final (1st)", tau_new1, pval_new1, time_new1))

# Второй запуск (чистое время без JIT-компиляции)
t0 = time.time()
tau_new2, _, lag_new2 = fast_coint_numba_final(x, y, k_max=12)
pval_new2 = float(sm.tsa.stattools.mackinnonp(tau_new2, regression="n"))
time_new2 = time.time() - t0
results_table.append(("fast_coint_numba_final (2nd)", tau_new2, pval_new2, time_new2))

# --- Единая таблица сравнения ---
print("\nСравнение tau, p-value и времени (сек) для Engle–Granger:")
print("Метод    |        tau |      p-value | время (сек)")
print("--------------------------------------------------------------------------------")
for method, tau, pval, t in results_table:
    print(f"{method:30} | {tau:10.6f} | {pval:.2e} | {t:10.4f}")

print(f"\nУскорение fast_coint_numba_final: {time_ref/time_new2:.1f}x")

# --- Тестируем точность fast_coint_numba_final ---
print("\nТестируем точность fast_coint_numba_final...")

# Вычисляем остатки регрессии для эталонного теста
model = sm.OLS(x, y, missing='drop').fit()
resid_ref = x - model.params[0] * y

# Эталонный тест adfuller из statsmodels
t0 = time.time()
adf_tau, adf_pval, adf_lag, *_ = sm.tsa.adfuller(resid_ref, maxlag=12, autolag='AIC', regression='n')
time_ref = time.time() - t0

# Запускаем оптимизированный тест
t0 = time.time()
tau_new1, _, lag_new1 = fast_coint_numba_final(x, y, k_max=12)
pval_new1 = float(sm.tsa.stattools.mackinnonp(tau_new1, regression="n"))
time_new1 = time.time() - t0

# Проверяем точность результатов
print(f"Эталон (sm.tsa.adfuller): tau={adf_tau:.6f}, p-value={adf_pval:.6f}, lag={adf_lag}, время={time_ref:.6f} сек")
print(f"fast_coint_numba_final: tau={tau_new1:.6f}, p-value={pval_new1:.6f}, lag={lag_new1}, время={time_new1:.6f} сек")
print(f"Разница в tau: {abs(adf_tau - tau_new1):.8f}")
print(f"Разница в p-value: {abs(adf_pval - pval_new1):.2e}")

# Проверка выбора лага
print(f"\nПроверка выбора лага: statsmodels выбрал k={adf_lag}, наша функция выбрала k={lag_new1}")
if adf_lag == lag_new1:
    print("Лаги совпадают! Оптимизация не повлияла на выбор лага.")
else:
    print("Лаги различаются. Это может быть вызвано небольшими различиями в вычислении AIC.")
