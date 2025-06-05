import numpy as np
import numba
from numba import njit # Для краткости
import statsmodels.tsa.stattools as smtsa

# --- Внутренние Numba-функции для расчета коинтеграции ---

@njit(cache=True, fastmath=True, nogil=True, error_model='numpy')
def _precompute_differences_numba(y: np.ndarray, maxlag: int) -> np.ndarray:
    """Предварительное вычисление разностей для ускорения ADF-теста."""
    n = y.shape[0]
    # Убедимся, что на выходе будет хотя бы пустой массив, если n-1 < 0
    # Это важно, если y имеет 0 или 1 элемент.
    # В Numba, если n-1 отрицательное, это может вызвать ошибку при создании dy.
    # Однако, если n=0, y[0] вызовет ошибку. Если n=1, n-1=0, dy будет пустым.
    if n <= 1:
        return np.empty(0, dtype=np.float64)
    dy = np.empty(n - 1, dtype=np.float64) # Используем empty вместо zeros для скорости
    for i in range(n - 1):
        dy[i] = y[i + 1] - y[i]
    return np.ascontiguousarray(dy) # Гарантируем непрерывность памяти


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy')
def _adf_autolag_fixed_lag_numba(res: np.ndarray, du_precomputed: np.ndarray, k_max_for_diff: int) -> tuple[float, int, int]:
    """
    ADF-подобный тест с фиксированным лагом k=2 для остатков коинтеграционной регрессии.
    k_max_for_diff используется для согласования с размером du_precomputed, но лаг ADF здесь всегда 2.
    Возвращает tau-статистику, использованный лаг (k=2) и эффективное количество наблюдений в ADF регрессии.
    """
    # Статические буферы для k=2 (m=3 регрессора: res_{t-1}, diff_res_{t-1}, diff_res_{t-2})
    e0 = np.array([1.0, 0.0, 0.0], dtype=np.float64) # Для извлечения var(beta_0)
    z_buf = np.empty(3, dtype=np.float64)
    w_buf = np.empty(3, dtype=np.float64)
    beta_coeffs_buf = np.empty(3, dtype=np.float64)

    n_orig_res = res.size
    k_adf_lag = 2 # Фиксированный лаг для этого теста

    # n_eff_adf - количество наблюдений в ADF-регрессии
    # y_adf = res[k+1:] - res[k:-1] (это du_precomputed[k:])
    # X_adf состоит из res[k:-1], du_precomputed[k-1:-1], ..., du_precomputed[0:-(k)]
    # Для k=2: y_adf = du_precomputed[2:]
    # X_adf: res[2:-1], du_precomputed[1:-2], du_precomputed[0:-3]
    # Размер y_adf: n_orig_res - 1 (для du) - k_adf_lag = n_orig_res - 3
    n_eff_adf = n_orig_res - 1 - k_adf_lag 

    min_obs_adf = 10 # Минимальное количество наблюдений для ADF регрессии
    if n_eff_adf < min_obs_adf:
        return np.nan, k_adf_lag, n_eff_adf # Возвращаем NaN tau, если данных мало

    # Формирование зависимой переменной и матрицы регрессоров для ADF
    # y_adf_target: diff(res_{t-1}) = res_t - res_{t-1}, лагированные на k_adf_lag
    # То есть, это du_precomputed[k_adf_lag:], например, du_precomputed[2:]
    y_adf_target = du_precomputed[k_adf_lag : k_adf_lag + n_eff_adf]

    # Матрица регрессоров X_adf_regressors [n_eff_adf, num_regressors_adf]
    # num_regressors_adf = 1 (для res_{t-1}) + k_adf_lag (для лагированных разностей)
    num_regressors_adf = 1 + k_adf_lag # =3 для k_adf_lag=2
    X_adf_regressors = np.empty((n_eff_adf, num_regressors_adf), dtype=np.float64)

    # 1. Лагированный уровень остатков: res[k_adf_lag : n_orig_res - 1]
    X_adf_regressors[:, 0] = res[k_adf_lag : n_orig_res - 1]

    # 2. Лагированные разности остатков
    # Для k_adf_lag=2: du_precomputed[1:n_eff_adf+1], du_precomputed[0:n_eff_adf]
    for i in range(k_adf_lag):
        # du_precomputed[ (k_adf_lag-1-i) : (k_adf_lag-1-i) + n_eff_adf ]
        X_adf_regressors[:, i + 1] = du_precomputed[(k_adf_lag - 1 - i) : (k_adf_lag - 1 - i) + n_eff_adf]

    # Решение нормальных уравнений X'X * b = X'y через разложение Холецкого
    xtx = X_adf_regressors.T @ X_adf_regressors
    xty = X_adf_regressors.T @ y_adf_target

    try:
        L = np.linalg.cholesky(xtx)
    except np.linalg.LinAlgError: # Матрица может быть не положительно определена
        return np.nan, k_adf_lag, n_eff_adf

    # Решаем L * z_buf = xty  => z_buf = L^{-1} * xty
    # Решаем L' * beta_coeffs_buf = z_buf => beta_coeffs_buf = (L')^{-1} * z_buf
    # np.linalg.solve(A,B) решает A*x=B
    try:
        z_buf[:] = np.linalg.solve(L, xty)
        beta_coeffs_buf[:] = np.linalg.solve(L.T, z_buf)
    except np.linalg.LinAlgError:
        return np.nan, k_adf_lag, n_eff_adf # Сингулярная система

    # Вычисление суммы квадратов остатков (SSR) для ADF регрессии
    # SSR = y'y - b'X'y
    yty = y_adf_target @ y_adf_target
    beta_xty = beta_coeffs_buf @ xty
    ssr = yty - beta_xty

    df_resid_adf = n_eff_adf - num_regressors_adf
    if df_resid_adf <= 0:
        return np.nan, k_adf_lag, n_eff_adf

    sigma2_adf = ssr / df_resid_adf
    if sigma2_adf < 1e-12: # Если дисперсия остатков очень мала или отрицательна
        return np.nan, k_adf_lag, n_eff_adf

    # Вычисление стандартной ошибки для коэффициента при res_{t-1} (первый регрессор)
    # se(b_0) = sqrt(sigma2_adf * (X'X)^{-1}_{0,0})
    # (X'X)^{-1}_{0,0} = (L^{-1}_{0,:}) @ (L^{-1}_{0,:}).T  = w_buf[0] where L'w = e0
    try:
        z_buf[:] = np.linalg.solve(L, e0) # L * z = e0
        w_buf[:] = np.linalg.solve(L.T, z_buf) # L'* w = z
    except np.linalg.LinAlgError:
        return np.nan, k_adf_lag, n_eff_adf

    xtx_inv_00 = w_buf[0]
    var_beta0 = sigma2_adf * xtx_inv_00

    if var_beta0 < 1e-12 or np.isnan(var_beta0): # Если дисперсия коэффициента мала, ноль или NaN
        tau_stat = np.nan
    else:
        se_beta0 = np.sqrt(var_beta0)
        if se_beta0 < 1e-9: # Если SE очень мало
             tau_stat = np.nan if np.abs(beta_coeffs_buf[0]) < 1e-9 else beta_coeffs_buf[0] / 1e-9 # Избегаем деления на ноль
        else:
            tau_stat = beta_coeffs_buf[0] / se_beta0

    return tau_stat, k_adf_lag, n_eff_adf


@njit(cache=True, fastmath=True, nogil=True, error_model='numpy')
def _fast_coint_beta_tau_k_numba(series_dep: np.ndarray, series_indep: np.ndarray, k_max_adf_lags_for_diff: int) -> tuple[float, float, int, int]:
    """
    Вычисляет коэффициент beta для регрессии series_dep = beta * series_indep + resid,
    и tau-статистику ADF-теста (с фикс. лагом k=2) для остатков этой регрессии.

    Args:
        series_dep (np.ndarray): Зависимый временной ряд (например, log_price_Y).
        series_indep (np.ndarray): Независимый временной ряд (например, log_price_X).
        k_max_adf_lags_for_diff (int): Максимальный лаг для precompute_differences.
                                     ADF-тест на остатках использует фиксированный лаг k=2.

    Returns:
        tuple[float, float, int, int]: (beta, tau_statistic, adf_lag_used (всегда 2), n_eff_adf).
    """
    n = series_dep.size
    if n == 0 or series_indep.size != n:
        return np.nan, np.nan, 2, 0 # k=2 как значение по умолчанию для лага, 0 для n_eff_adf

    # Регрессия series_dep на series_indep: series_dep_i = beta * series_indep_i + resid_i
    # beta = sum(series_dep_i * series_indep_i) / sum(series_indep_i * series_indep_i)
    # Это если регрессия без константы. Для коинтеграции обычно регрессия с константой,
    # но остатки потом центрируются или тест ADF учитывает это.
    # Стандартный тест Энгла-Грейнджера: OLS y на x (с константой), затем ADF на остатках.
    # Здесь OLS без константы, что может быть не совсем стандартно для EG, но ADF на остатках.
    # Однако, если ряды предварительно центрированы (например, вычтен средний лог цены), то это ок.
    # Для большей общности, лучше использовать OLS с константой или убедиться, что ряды центрированы.
    # Пока оставим как есть, предполагая, что это соответствует предыдущей логике.
    
    # Расчет beta (без константы)
    sum_dep_indep = 0.0
    sum_indep_sq = 0.0
    for i in range(n):
        sum_dep_indep += series_dep[i] * series_indep[i]
        sum_indep_sq += series_indep[i] * series_indep[i]

    if sum_indep_sq == 0.0:
        beta = np.nan
    else:
        beta = sum_dep_indep / sum_indep_sq

    # Вычисление остатков
    resid = np.empty(n, dtype=np.float64)
    if np.isnan(beta):
        # Если beta не определен, остатки тоже не определены.
        # Для ADF теста на таких остатках результат будет некорректным.
        return beta, np.nan, 2, 0 # k=2 как значение по умолчанию, 0 для n_eff_adf
    else:
        for i in range(n):
            resid[i] = series_dep[i] - beta * series_indep[i]

    # ADF тест на остатках
    # Убедимся, что достаточно данных для precompute_differences и последующего ADF
    # _precompute_differences_numba требует n > 1.
    # _adf_autolag_fixed_lag_numba требует n_orig_res - 1 - k_adf_lag >= min_obs_adf
    # где n_orig_res = len(resid), k_adf_lag = 2. Так что len(resid) >= min_obs_adf + 3.
    min_len_for_adf = 10 + 3 # Примерно
    if n < min_len_for_adf: 
        return beta, np.nan, 2, 0 # 0 для n_eff_adf

    du_precomputed_resid = _precompute_differences_numba(resid, k_max_adf_lags_for_diff)
    if du_precomputed_resid.size == 0 and n > 1 : # Если resid был, но разности не получились (например, resid был из 1 элемента)
         return beta, np.nan, 2, 0 # 0 для n_eff_adf
    if du_precomputed_resid.size < k_max_adf_lags_for_diff and n > k_max_adf_lags_for_diff +1 : # если разностей меньше чем ожидалось
        # это может произойти если resid был слишком коротким для k_max_adf_lags_for_diff
        # но достаточно длинным для ADF с k=2. k_max_adf_lags_for_diff здесь больше для выделения памяти.
        pass # Продолжаем, _adf_autolag_fixed_lag_numba сам проверит длину du_precomputed

    tau_stat, k_lag_used, n_eff_adf = _adf_autolag_fixed_lag_numba(resid, du_precomputed_resid, k_max_adf_lags_for_diff)
195: 
196:     return beta, tau_stat, k_lag_used, n_eff_adf




def calculate_adf_pvalue(
    tau_stat: float, 
    n_eff_adf: int, 
    regression_type: str = 'c'  # 'c', 'nc', 'ct', 'ctt'
) -> float:
    """
    Calculates the p-value for an ADF test given the tau-statistic,
    effective number of observations, and regression type using mackinnonp.

    Args:
        tau_stat (float): The tau-statistic from the ADF test.
        n_eff_adf (int): The effective number of observations used in the ADF regression.
        regression_type (str): The type of regression used in the ADF test.
                               ('c' for constant, 'nc' for no constant, 'ct' for constant and trend,
                                'ctt' for constant, trend, and trend squared).

    Returns:
        float: The calculated p-value, or np.nan if calculation fails.
    """
    if np.isnan(tau_stat) or n_eff_adf == 0:
        return np.nan
    try:
        # For ADF test on a single series, N=1.
        # nobs is the effective number of observations.
        p_val = smtsa.mackinnonp(tau_stat, regression=regression_type, N=1, nobs=n_eff_adf)
        # Handle cases where mackinnonp might return extreme values outside [0,1]
        # or if tau is too extreme for the tables.
        if p_val < 0.0: p_val = 0.0
        if p_val > 1.0: p_val = 1.0
        return float(p_val)
    except (ValueError, FloatingPointError, IndexError) as e:
        # Handle errors from mackinnonp (e.g., tau outside table range)
        # logging.warning(f"Mackinnonp error for tau={tau_stat}, nobs={n_eff_adf}, reg='{regression_type}': {e}")
        # Fallback for extreme tau values not covered by tables
        if tau_stat < -20: # Arbitrary very small tau
            return 0.0
        elif tau_stat > 5: # Arbitrary very large tau
            return 1.0
        return np.nan


# --- Публичная Python-функция для расчета параметров коинтеграции ---

def calculate_coint_params(series_dep: np.ndarray, 
                           series_indep: np.ndarray, 
                           k_max_adf_lags_for_diff: int, 
                           regression_coint_trend: str = 'c'
                           ) -> tuple[float, float, float, int]:
    """
    Рассчитывает параметры коинтеграции для двух временных рядов.
    Использует Numba-ускоренное ядро для вычисления beta и tau-статистики,
    затем вычисляет p-value с помощью statsmodels.mackinnonp.

    Регрессия для beta: series_dep = beta * series_indep + residuals.
    ADF-тест на остатках (тип регрессии 'c' по умолчанию для mackinnonp, N=2).

    Args:
        series_dep (np.ndarray): Зависимый временной ряд.
        series_indep (np.ndarray): Независимый временной ряд.
        k_max_adf_lags_for_diff (int): Максимальный лаг для этапа precompute_differences в Numba.
                                     ADF-тест на остатках использует фиксированный лаг k=2.
        regression_coint_trend (str): Тип тренда для p-value теста коинтеграции ('c', 'ct', 'ctt', 'nc').
                                      Передается в mackinnonp. 'c' (константа) по умолчанию.

    Returns:
        tuple[float, float, float, int]: (beta, tau_statistic, p_value, adf_lag_used (всегда 2)).
    """
    # Проверка на пустые или несовпадающие по длине массивы
    if series_dep.size == 0 or series_dep.size != series_indep.size:
        return np.nan, np.nan, np.nan, 2

    # Центрируем ряды перед передачей в Numba функцию
    # y_centered = y - y_mean
    # x_centered = x - x_mean
    # Регрессия y_centered = beta * x_centered эквивалентна y = alpha + beta * x, 
    # где alpha = y_mean - beta * x_mean. Нам нужен только beta.
    series_dep_mean = np.mean(series_dep)
    series_indep_mean = np.mean(series_indep)
    
    series_dep_centered = series_dep - series_dep_mean
    series_indep_centered = series_indep - series_indep_mean
        
    beta, tau, k_lag, n_eff_adf = _fast_coint_beta_tau_k_numba(series_dep_centered, series_indep_centered, k_max_adf_lags_for_diff)

    if np.isnan(tau) or n_eff_adf == 0:
        p_value = np.nan
    else:
        try:
            # N=1, так как это ADF-тест на одномерном ряде остатков.
            # nobs - эффективное число наблюдений в ADF-регрессии.
            p_value = smtsa.mackinnonp(tau, regression=regression_coint_trend, N=1, nobs=n_eff_adf)
        except (ValueError, FloatingPointError, IndexError) as e:
            # mackinnonp может выбросить исключение для экстремальных значений tau или если tau не в таблице
            # IndexError может возникнуть, если tau слишком мал/велик для интерполяции
            # logger.debug(f"Error in mackinnonp for tau={tau}, trend={regression_coint_trend}: {e}") # Логгер здесь недоступен
            p_value = np.nan 
            if tau < -100: # Очень сильная стационарность, p-value должно быть близко к 0
                p_value = 0.0
            elif tau > 100: # Очень сильная нестационарность, p-value должно быть близко к 1
                p_value = 1.0

    return beta, tau, p_value, k_lag

# --- Numba-функции для общего ADF-теста (не для коинтеграции напрямую) ---

@numba.njit(cache=True)
def _numba_ols_for_adf_internal(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """
    Выполняет OLS регрессию y на X с использованием np.linalg.lstsq.
    Возвращает: (params, rss, t_stat_first_regressor, stderr_first_regressor, n_obs_reg)
    t_stat_first_regressor и stderr_first_regressor относятся к первому регрессору в X (лагированный уровень).
    """
    n_obs_reg = X.shape[0]
    k_params = X.shape[1]

    if n_obs_reg < k_params: # Недостаточно данных
        return (np.full(k_params, np.nan),
                np.nan,
                np.nan,
                np.nan,
                float(n_obs_reg))

    try:
        coeffs, residuals_sum_sq_arr, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        rss = np.nan
        if residuals_sum_sq_arr.size > 0:
            rss = residuals_sum_sq_arr[0]
        elif rank == k_params : # Если lstsq не вернула rss, но ранг полный, можем посчитать вручную
             rss = np.sum((y - X @ coeffs)**2)


        if rank < k_params: # Мультиколлинеарность
             return (coeffs, 
                    rss if not np.isnan(rss) else np.nan, # rss может быть вычислено lstsq даже при неполном ранге
                    np.nan,
                    np.nan,
                    float(n_obs_reg))

        if n_obs_reg <= k_params: # Степени свободы <= 0
            return (coeffs,
                    rss,
                    np.nan, # t_stat
                    np.nan, # stderr
                    float(n_obs_reg))
        
        # Если rss все еще nan, а должен быть (например, lstsq не вернула, но rank полный)
        if np.isnan(rss) and rank == k_params:
            rss = np.sum((y - X @ coeffs)**2) # Пересчитываем, если необходимо
        
        if np.isnan(rss): # Если rss так и не удалось определить
            return coeffs, np.nan, np.nan, np.nan, float(n_obs_reg)


        df_residuals = n_obs_reg - k_params
        mse_residuals = rss / df_residuals
        
        try:
            # X.T @ X может быть сингулярной или плохо обусловленной
            # Псевдообратная матрица более устойчива: np.linalg.pinv(X.T @ X)
            # Однако, для Numba, np.linalg.inv проще, если матрица хорошая.
            # Попробуем с проверкой на сингулярность через условие матрицы X.T @ X
            xtx = X.T @ X
            # Проверка на сингулярность по определителю (не очень надежно) или cond
            # В Numba нет np.linalg.cond, используем try-except для inv
            xtx_inv = np.linalg.inv(xtx)
            var_coeff_first = mse_residuals * xtx_inv[0, 0]
            if var_coeff_first < 0 or np.isnan(var_coeff_first): 
                stderr_first = np.nan
            else:
                stderr_first = np.sqrt(var_coeff_first)
        except np.linalg.LinAlgError: 
            stderr_first = np.nan

        if np.isnan(stderr_first) or stderr_first == 0.0:
            t_stat_first = np.nan
        else:
            # coeffs[0] это коэффициент при y_{t-1} (первый столбец в X_regressors_final)
            t_stat_first = coeffs[0] / stderr_first 

        return coeffs, rss, t_stat_first, stderr_first, float(n_obs_reg)

    except np.linalg.LinAlgError: 
        return (np.full(k_params, np.nan),
                np.nan,
                np.nan,
                np.nan,
                float(n_obs_reg))

@numba.njit(cache=True)
def _prepare_adf_regressors_numba(series: np.ndarray, k_lags: int, regression: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Подготавливает данные для ADF регрессии.
    Возвращает: (y_diff, X_regressors, n_obs_effective)
    Первый столбец в X_regressors - это лагированный уровень series_{t-1}.
    """
    n_total = len(series)
    start_idx = k_lags + 1 
    
    if n_total <= start_idx:
        return np.empty((0,), dtype=series.dtype), np.empty((0,0), dtype=series.dtype), 0

    y_diff = series[start_idx:] - series[start_idx-1:-1] 
    n_obs_effective = len(y_diff)

    if n_obs_effective == 0:
        return np.empty((0,), dtype=series.dtype), np.empty((0,0), dtype=series.dtype), 0

    series_lagged_level = series[k_lags : n_total-1][:n_obs_effective]
    
    num_deterministic_terms = 0
    if regression == 'c': num_deterministic_terms = 1
    elif regression == 'ct': num_deterministic_terms = 2
    elif regression == 'ctt': num_deterministic_terms = 3 # Нестандартный тип
        
    total_cols_in_X = 1 + k_lags + num_deterministic_terms
    X_regressors_final = np.empty((n_obs_effective, total_cols_in_X), dtype=series.dtype)
    
    X_regressors_final[:, 0] = series_lagged_level

    current_col_final = 1
    for lag_i in range(1, k_lags + 1): 
        lagged_diff_col = (series[start_idx-lag_i : n_total-lag_i] - 
                           series[start_idx-lag_i-1 : n_total-lag_i-1])
        X_regressors_final[:, current_col_final] = lagged_diff_col[:n_obs_effective]
        current_col_final +=1

    if regression == 'c':
        X_regressors_final[:, current_col_final] = 1.0
    elif regression == 'ct':
        X_regressors_final[:, current_col_final] = 1.0
        # Трендовый член должен соответствовать временным индексам y_diff
        # Если y_diff[0] это series[start_idx] - series[start_idx-1], то t = start_idx
        X_regressors_final[:, current_col_final + 1] = np.arange(start_idx, start_idx + n_obs_effective, dtype=series.dtype)
    elif regression == 'ctt': # Нестандартный тип
        X_regressors_final[:, current_col_final] = 1.0
        trend_vals = np.arange(start_idx, start_idx + n_obs_effective, dtype=series.dtype)
        X_regressors_final[:, current_col_final + 1] = trend_vals
        X_regressors_final[:, current_col_final + 2] = trend_vals**2
        
    return y_diff, X_regressors_final, n_obs_effective

@numba.njit(cache=True)
def fast_adf_test_numba(series: np.ndarray, maxlag: int = 0, regression: str = 'c', autolag: str | None = 'AIC') -> tuple[float, float, int, int, np.ndarray, float]:
    """
    Тест Дики-Фуллера с использованием Numba.
    Рассчитывает ADF-статистику, p-value, оптимальное число лагов (если autolag), число наблюдений.
    Critical_values НЕ рассчитываются и возвращаются как массив NaN.

    Параметры:
        series (np.ndarray): Временной ряд для тестирования.
        maxlag (int): Максимальное число лагов для включения.
        regression (str): Тип регрессии: 'c', 'nc', 'ct', 'ctt'.
        autolag (str | None): Метод выбора числа лагов: 'AIC', 'BIC', или None.

    Возвращает:
        adf_stat (float): ADF t-статистика.
        p_value (float): Рассчитанное p-value.
        usedlag (int): Число использованных лагов.
        nobs (int): Число наблюдений в ADF регрессии.
        critical_values_array (np.ndarray): np.array([np.nan, np.nan, np.nan]).
        icbest (float): Значение инф. критерия для `usedlag` (np.nan если autolag=None).
    """
    n_total = len(series)
    min_obs_for_test_heuristic = 10 
    
    num_deterministic_terms = 0
    if regression == 'nc': num_deterministic_terms = 0
    elif regression == 'c': num_deterministic_terms = 1
    elif regression == 'ct': num_deterministic_terms = 2
    elif regression == 'ctt': num_deterministic_terms = 3
    else: # Неизвестный тип регрессии
        return np.nan, np.nan, 0, n_total, np.array([np.nan, np.nan, np.nan]), np.nan

    # Максимально возможное число лагов, чтобы оставалось достаточно данных для регрессии
    # n_obs_effective = n_total - (k_lags + 1)
    # Число параметров = (1 для y_lag) + k_lags + num_deterministic_terms
    # Требуется: n_obs_effective > Число параметров
    # n_total - k_lags - 1 > 1 + k_lags + num_deterministic_terms
    # n_total - 2 - num_deterministic_terms > 2 * k_lags
    max_allowable_lags = (n_total - 2 - num_deterministic_terms) // 2
    if max_allowable_lags < 0: 
        max_allowable_lags = 0
    
    if maxlag > max_allowable_lags:
        maxlag = max_allowable_lags
    if maxlag < 0: # Если n_total очень мал
        maxlag = 0

    # Проверка, достаточно ли данных даже для 0 лагов
    # n_total - 1 > 1 + num_deterministic_terms (т.е. n_obs_eff для k_lags=0 > 1+num_det)
    if (n_total - 1) <= (1 + num_deterministic_terms):
         return np.nan, np.nan, 0, (n_total - 1 if n_total > 0 else 0), np.array([np.nan, np.nan, np.nan]), np.nan

    adf_stat = np.nan
    p_value = np.nan # Будет вычислено позже
    usedlag = 0
    nobs_adf_reg = 0
    # critical_values_array не вычисляются здесь, как и ранее
    critical_values_array = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    icbest = np.nan

    if autolag in ['AIC', 'BIC']:
        best_ic_val = np.inf
        best_lag_val = 0 # По умолчанию 0 лагов, если ни один не подходит

        # Проверяем лаг 0 отдельно, чтобы установить базовый лучший лаг
        y_diff_0, X_reg_0, n_obs_reg_0 = _prepare_adf_regressors_numba(series, 0, regression)
        if n_obs_reg_0 >= min_obs_for_test_heuristic and X_reg_0.shape[0] > 0 and X_reg_0.shape[1] > 0 and n_obs_reg_0 > X_reg_0.shape[1]:
            _, rss_0, _, _, _ = _numba_ols_for_adf_internal(y_diff_0, X_reg_0)
            if not np.isnan(rss_0) and rss_0 > 1e-9: # 1e-9 для избежания log(0)
                num_params_0 = X_reg_0.shape[1]
                log_rss_term_0 = n_obs_reg_0 * np.log(rss_0 / n_obs_reg_0)
                if autolag == 'AIC':
                    current_ic_0 = log_rss_term_0 + 2 * num_params_0
                else: # BIC
                    current_ic_0 = log_rss_term_0 + np.log(n_obs_reg_0) * num_params_0
                if current_ic_0 < best_ic_val:
                    best_ic_val = current_ic_0
                    best_lag_val = 0

        for k_lags_candidate in range(1, maxlag + 1): # Начинаем с 1, т.к. 0 уже мог быть проверен
            y_diff, X_reg, n_obs_reg = _prepare_adf_regressors_numba(series, k_lags_candidate, regression)
            
            if n_obs_reg < min_obs_for_test_heuristic or X_reg.shape[0] == 0 or X_reg.shape[1] == 0 or n_obs_reg <= X_reg.shape[1]:
                continue

            _, rss, _, _, _ = _numba_ols_for_adf_internal(y_diff, X_reg)

            if np.isnan(rss) or rss < 1e-9: 
                current_ic = np.inf
            else:
                num_params = X_reg.shape[1]
                log_rss_term = n_obs_reg * np.log(rss / n_obs_reg)
                if autolag == 'AIC':
                    current_ic = log_rss_term + 2 * num_params
                else: # BIC
                    current_ic = log_rss_term + np.log(n_obs_reg) * num_params
            
            if current_ic < best_ic_val:
                best_ic_val = current_ic
                best_lag_val = k_lags_candidate
        
        usedlag = best_lag_val
        icbest = best_ic_val

    else: # autolag is None, используем maxlag как usedlag
        usedlag = maxlag
        # icbest остается np.nan

    # Финальный расчет ADF статистики с выбранным usedlag
    y_diff_final, X_regressors_final, nobs_adf_reg = _prepare_adf_regressors_numba(series, usedlag, regression)
    
    if nobs_adf_reg < min_obs_for_test_heuristic or X_regressors_final.shape[0] == 0 or X_regressors_final.shape[1] == 0 or nobs_adf_reg <= X_regressors_final.shape[1]:
        # Если даже с выбранным usedlag данных недостаточно, возвращаем NaN
        return np.nan, p_value, usedlag, nobs_adf_reg, critical_values_array, icbest

    coeffs_final, _, adf_stat, _, _ = _numba_ols_for_adf_internal(y_diff_final, X_regressors_final)
    
    # p_value и critical_values не рассчитываются здесь,    # Вычисляем p-value перед возвратом
    if not np.isnan(adf_stat) and nobs_adf_reg > 0:
        p_value = calculate_adf_pvalue(adf_stat, nobs_adf_reg, regression)
    
    # Возвращаем результаты
    return adf_stat, p_value, usedlag, nobs_adf_reg, critical_values_array, icbest


# --- Функции для работы с ценовыми данными пар, расчет параметров коинтеграции ---

# На этом месте будет функция compute_coint_and_beta, перенесенная из backtest.py

from typing import Any, Dict, Optional, Protocol, Tuple
from datetime import datetime
from functools import lru_cache


class BacktestParamsProtocol(Protocol):
    """Protocol for BacktestParams to avoid circular imports."""
    adf_max_lag: int
    tau_open_threshold: float
    min_data_points_for_coint: int


@lru_cache(maxsize=128)
def compute_coint_and_beta(
    y_series: np.ndarray,
    x_series: np.ndarray,
    params: BacktestParamsProtocol,
    regression_coint_trend: str = 'c'
) -> Dict[str, Any]:
    """
    Рассчитывает различные метрики коинтеграции для пары рядов.

    Включает: бета, тау-статистику, проверку на коинтеграцию (is_coint),
    количество наблюдений (n_obs), p-value и оптимальный лаг.

    Args:
        y_series (np.ndarray): Зависимый временной ряд (обычно логарифмические цены первого инструмента).
        x_series (np.ndarray): Независимый временной ряд (обычно логарифмические цены второго инструмента).
        params (BacktestParamsProtocol): Параметры для расчета коинтеграции.
        regression_coint_trend (str): Тип регрессии для ADF-теста на остатках.

    Returns:
        Dict[str, Any]: Словарь с результатами.
        Ключи словаря: 'beta', 'tau_stat', 'is_coint', 'n_obs', 'adf_pvalue', 'optimal_lag', 'error_msg'.
    """
    result: Dict[str, Any] = {
        'beta': None,
        'tau_stat': None,
        'is_coint': False,
        'n_obs': len(y_series),
        'adf_pvalue': None,
        'optimal_lag': None,
        'error_msg': None,
    }

    if len(y_series) < params.min_data_points_for_coint:
        result['error_msg'] = f"Insufficient data points ({len(y_series)}) for cointegration. Need {params.min_data_points_for_coint}."
        return result

    try:
        # Вызываем calculate_coint_params
        beta, tau_stat, p_value, adf_lag = calculate_coint_params(
            series_dep=y_series, 
            series_indep=x_series,
            k_max_adf_lags_for_diff=params.adf_max_lag,
            regression_coint_trend=regression_coint_trend
        )

        result['beta'] = beta
        result['tau_stat'] = tau_stat
        result['adf_pvalue'] = p_value
        result['optimal_lag'] = adf_lag

        if np.isnan(beta) or np.isnan(tau_stat):
            result['error_msg'] = f"calculate_coint_params returned NaN for beta/tau"
            result['is_coint'] = False
            return result
        
        # Определяем is_coint на основе tau_stat (более отрицательный tau = более сильная коинтеграция)
        if tau_stat is not None and not np.isnan(tau_stat):
            result['is_coint'] = tau_stat < params.tau_open_threshold
        else:
            result['is_coint'] = False
        
        result['beta'] = float(result['beta'])
        result['tau_stat'] = float(result['tau_stat'])

        return result

    except Exception as e:
        result['error_msg'] = f"Error calculating cointegration: {str(e)}"
        return result


def calculate_pair_beta_and_tau_static(
    series_dep: np.ndarray,
    series_indep: np.ndarray,
    min_data_points: int = 30,
    max_adf_lag: int = 2
) -> tuple[float | None, float | None]:
    """
    Рассчитывает бету и тау-статистику для пары временных рядов.
    
    Упрощенная версия compute_coint_and_beta, возвращающая только бету и тау-статистику.
    Использует calculate_coint_params для расчетов.

    Args:
        series_dep (np.ndarray): Зависимый временной ряд.
        series_indep (np.ndarray): Независимый временной ряд.
        min_data_points (int): Минимальное количество точек данных для расчета.
        max_adf_lag (int): Максимальное количество лагов для ADF-теста.

    Returns:
        tuple[float | None, float | None]: Кортеж (beta, tau_statistic).
                                          Возвращает (None, None) в случае ошибки или недостатка данных.
    """
    # Проверка количества точек данных
    if len(series_dep) < min_data_points or len(series_indep) < min_data_points:
        return None, None

    try:
        # Вызываем calculate_coint_params
        beta, tau_stat, _p_value, _adf_lag = calculate_coint_params(
            series_dep=series_dep, 
            series_indep=series_indep,
            k_max_adf_lags_for_diff=max_adf_lag,
            regression_coint_trend='c'  # 'c' для ADF на остатках с константой
        )
        
        if beta is not None and not np.isnan(beta) and tau_stat is not None and not np.isnan(tau_stat):
            return float(beta), float(tau_stat)
        else:
            return None, None

    except Exception:
        return None, None
