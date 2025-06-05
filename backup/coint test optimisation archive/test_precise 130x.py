# Установка переменных окружения для многопоточности OpenBLAS
import os
# Отключаем многопоточность BLAS для избежания конфликтов с Numba
os.environ['OPENBLAS_NUM_THREADS'] = '1'  
os.environ['OMP_NUM_THREADS'] = '1'        
os.environ['MKL_NUM_THREADS'] = '1'

from numba import njit
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from numba import njit

# --- u0421u043eu0437u0434u0430u0451u043c u0442u0435u0441u0442u043eu0432u044bu0435 u0434u0430u043du043du044bu0435 ---
np.random.seed(42)
n = 5670

# u0414u0432u0430 u0441u043bu0443u0447u0430u0439u043du044bu0445 u0431u043bu0443u0436u0434u0430u043du0438u044f
x = np.random.normal(0, 1, n).cumsum()
y = np.random.normal(0, 1, n).cumsum()

# --- u041eu043fu0442u0438u043cu0438u0437u0438u0440u043eu0432u0430u043du043du0430u044f u0440u0435u0430u043bu0438u0437u0430u0446u0438u044f fast_coint_numba_new ---
@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
def _adf_autolag_numba(res, du, k_max):
    n = res.size
    best_aic = 1e300
    best_tau = 0.0
    best_k   = 0
    
    # Выделяем память для матрицы X максимального размера один раз
    max_n_eff = n - 1  # Максимальное число наблюдений (при k=0)
    # Не инициализируем матрицу нулями, т.к. все используемые ячейки будут перезаписаны
    X_max = np.empty((max_n_eff, k_max + 1), np.float64)
    # Делаем X_max непрерывным один раз (устраняет NumbaPerformanceWarning)
    X_max = np.ascontiguousarray(X_max)
    
    # Создаем единичный вектор e0 = [1,0,...,0] максимального размера для вычисления (XᵀX)⁻¹[0,0]
    e0_max = np.empty(k_max + 1, np.float64)
    # Делаем e0_max непрерывным один раз
    e0_max = np.ascontiguousarray(e0_max)
    # Заполняем вектор через одномерный цикл
    for i in range(k_max + 1):
        e0_max[i] = 1.0 if i == 0 else 0.0
    
    # Предварительно вычисляем логарифмы для всех возможных значений nobs
    # nobs = n - k - 1, где k изменяется от 0 до k_max
    # Таким образом, nobs изменяется от n-1 до n-k_max-1
    precomputed_log_nobs = np.empty(k_max + 1, np.float64)
    # Делаем precomputed_log_nobs непрерывным один раз
    precomputed_log_nobs = np.ascontiguousarray(precomputed_log_nobs)
    # Заполняем и вычисляем через одномерный цикл
    for k in range(k_max + 1):
        nobs_k = n - k - 1
        precomputed_log_nobs[k] = np.log(nobs_k)
    

    
    # Точно соответствуем реализации statsmodels
    for k in range(k_max + 1):
        n_eff = n - k - 1
        if n_eff < 10:
            continue
        
        m = k + 1
        y = du[k:]
        # Используем часть предварительно выделенной матрицы
        X = X_max[:n_eff, :m]
        # Срезы могут нарушать непрерывность, поэтому всё же нужна копия
        X = np.ascontiguousarray(X)
        X[:, 0] = res[k:-1]
        
        # Формируем матрицу регрессоров точно как в statsmodels
        for j in range(k):
            X[:, j + 1] = du[k - j - 1 : n - j - 2]
            
        # Решаем нормальные уравнения через разложение Холецкого
        xtx = X.T @ X   # m×m симметричная положительно определённая
        xty = X.T @ y   # m
        # Разложение Холецкого быстрее на 20-25% для SPD-матриц
        L = np.linalg.cholesky(xtx)  # Нижняя треугольная матрица L
        # Решаем две треугольные системы: L z = xty, затем Lᵀ beta = z
        z = np.linalg.solve(L, xty)
        beta = np.linalg.solve(L.T, z)
        
        # Вычисляем сумму квадратов остатков без формирования вектора остатков
        # Используем тождество: SSR = yᵀy − βᵀXᵀy = y@y − beta.dot(xty)
        yty = y @ y  # Скалярное произведение
        ssr = yty - beta.dot(xty)  # Быстрее, чем формировать вектор остатков
        nobs = n_eff
        
        # AIC использует sigma2 = ssr / nobs (оценка МП)
        # Оптимизированная формула: log(ssr/nobs) + 2m/nobs = log(ssr) - log(nobs) + 2m/nobs
        # Используем предварительно вычисленные логарифмы для nobs
        aic = np.log(ssr) - precomputed_log_nobs[k] + 2.0 * m / nobs
        
        # Стандартная ошибка использует sigma2 = ssr / df_resid (несмещенная оценка)
        # m - количество регрессоров (k+1)
        df_resid = nobs - m
        if df_resid <= 0:
            # Если степеней свободы не осталось, этот лаг k невалиден.
            # Пропускаем его (или можно присвоить AIC очень большое значение).
            continue
        
        # Вычисляем sigma2 для стандартной ошибки
        sigma2_for_se = ssr / df_resid
        
        # Вычисляем только нужный элемент (0,0) матрицы (XᵀX)⁻¹
        # Используем уже вычисленное разложение Холецкого и предварительно созданный вектор
        e0 = e0_max[:m]  # Используем срез предварительно созданного вектора
        # Решаем две треугольные системы: L z = e0, затем Lᵀ w = z
        z = np.linalg.solve(L, e0)
        w = np.linalg.solve(L.T, z)  # w[0] = (XᵀX)⁻¹[0,0]
        
        # Вычисляем стандартную ошибку и t-статистику
        se_b0 = np.sqrt(sigma2_for_se * w[0])  # Используем только нужный элемент
        tau = beta[0] / se_b0
        
        if aic < best_aic:
            best_aic, best_tau, best_k = aic, tau, k
            
    return best_tau, best_k







@njit(fastmath=True, cache=True)
def fast_coint_numba_new(x, y, k_max=12):
    # Преобразуем входные данные в float64 для стабильности
    # В Numba можно сравнивать строковое представление dtype
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
    tau, k = _adf_autolag_numba(resid, du, k_max)
    
    return tau, None, k



# --- u0421u0440u0430u0432u043du0438u0432u0430u0435u043c u0440u0435u0430u043bu0438u0437u0430u0446u0438u0438 ---
results_table = []

# u0422u0435u0441u0442u0438u0440u0443u0435u043c statsmodels.coint
t0 = time.time()
tau_ref, pval_ref, _ = coint(x, y, trend='n')
time_ref = time.time() - t0
results_table.append(("statsmodels.coint", tau_ref, pval_ref, time_ref))

# u041fu0435u0440u0432u044bu0439 u0437u0430u043fu0443u0441u043a fast_coint_numba_new (u0432u043au043bu044eu0447u0430u044f JIT-u043au043eu043cu043fu0438u043bu044fu0446u0438u044e)
t0 = time.time()
tau_new1, _, lag_new1 = fast_coint_numba_new(x, y, k_max=12)
pval_new1 = float(sm.tsa.stattools.mackinnonp(tau_new1, regression="n"))
time_new1 = time.time() - t0
results_table.append(("fast_coint_numba_new (1st)", tau_new1, pval_new1, time_new1))

# u0412u0442u043eu0440u043eu0439 u0437u0430u043fu0443u0441u043a (u0447u0438u0441u0442u043eu0435 u0432u0440u0435u043cu044f u0431u0435u0437 JIT-u043au043eu043cu043fu0438u043bu044fu0446u0438u0438)
t0 = time.time()
tau_new2, _, lag_new2 = fast_coint_numba_new(x, y, k_max=12)
pval_new2 = float(sm.tsa.stattools.mackinnonp(tau_new2, regression="n"))
time_new2 = time.time() - t0
results_table.append(("fast_coint_numba_new (2nd)", tau_new2, pval_new2, time_new2))

# --- u0415u0434u0438u043du0430u044f u0442u0430u0431u043bu0438u0446u0430 u0441u0440u0430u0432u043du0435u043du0438u044f ---
import timeit

# Выводим результаты сравнения
print("\nu0421u0440u0430u0432u043du0435u043du0438u0435 tau, p-value u0438 u0432u0440u0435u043cu0435u043du0438 (u0441u0435u043a) u0434u043bu044f Engleu2013Granger:")
print("u041cu0435u0442u043eu0434    |        tau |      p-value | u0432u0440u0435u043cu044f (u0441u0435u043a)")
print("--------------------------------------------------------------------------------")
for method, tau, pval, t in results_table:
    print(f"{method:30} | {tau:10.6f} | {pval:.2e} | {t:10.4f}")

print(f"\nu0423u0441u043au043eu0440u0435u043du0438u0435 fast_coint_numba_new: {time_ref/time_new2:.1f}x")

# --- Тестируем точность fast_coint_numba_new ---
print("\nТестируем точность fast_coint_numba_new...")

# Генерируем данные для теста
np.random.seed(42)
x = np.random.normal(0, 1, n).cumsum()
y = np.random.normal(0, 1, n).cumsum()

# Максимальное количество лагов для теста
k_max_bench = 12

# Запускаем эталонный тест
t0 = time.time()
beta_ref, resid_ref = ols_beta_resid_no_const_new(x.astype(np.float64), y.astype(np.float64))
adf_tau, adf_pval, adf_lag, *_ = sm.tsa.adfuller(resid_ref, maxlag=k_max_bench, autolag='AIC', regression='n')
time_ref = time.time() - t0

# Запускаем оптимизированный тест
t0 = time.time()
tau_new1, _, lag_new1 = fast_coint_numba_new(x, y, k_max=k_max_bench)
pval_new1 = float(sm.tsa.stattools.mackinnonp(tau_new1, regression="n"))
time_new1 = time.time() - t0

# Проверяем точность результатов
print(f"Эталон (sm.tsa.adfuller): tau={adf_tau:.6f}, p-value={adf_pval:.6f}, lag={adf_lag}, время={time_ref:.6f} сек")
print(f"fast_coint_numba_new: tau={tau_new1:.6f}, p-value={pval_new1:.6f}, lag={lag_new1}, время={time_new1:.6f} сек")
print(f"Разница в tau: {abs(adf_tau - tau_new1):.8f}")
print(f"Разница в p-value: {abs(adf_pval - pval_new1):.2e}")

# Проверяем, что p-value обоих тестов > 0.1
assert adf_pval > 0.1 and pval_new1 > 0.1, "p-value всё ещё слишком маленькое"

# Проверяем, что разница в tau не превышает допустимый порог
assert abs(adf_tau - tau_new1) < 0.01, f"Разница в tau ({abs(adf_tau - tau_new1):.8f}) слишком большая"
