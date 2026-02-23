import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# ==============================
# 1. Descarga de la data
# ==============================

ticker = "BTC-USD"

data = yf.download(
    ticker,
    start="2020-01-01",
    end="2024-01-01",
    auto_adjust=True
)

# Simplificar columnas MultiIndex si es necesario
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
    print("Simplified columns:", data.columns)

# ==============================
# 2. Retornos Diarios
# ==============================

data["Returns"] = data["Close"].pct_change()

print("\nDaily Returns Summary:")
print(data["Returns"].describe())

print("\nFirst 5 rows:")
print("-----------------------------------")
print(data.head())
print("-----------------------------------")
print(data.info())

print("\nMissing values per column:")
print("-----------------------------------")
print(data.isnull().sum())
print("-----------------------------------")

# ==============================
# 3. Media Movil Simple (SMA)
# ==============================

data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

data["Signal"] = 0
data.loc[data["SMA_20"] > data["SMA_50"], "Signal"] = 1

data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]

# ==============================
# 4. Retornos Acumulados
# ==============================

data["Market_Cumulative"] = (1 + data["Returns"]).cumprod()
data["Strategy_Cumulative"] = (1 + data["Strategy_Returns"]).cumprod()

# ==============================
# 5. Métricas de Rendimiento
# ==============================

# Daily stats
market_daily_mean = data["Returns"].mean()
strategy_daily_mean = data["Strategy_Returns"].mean()

market_daily_vol = data["Returns"].std()
strategy_daily_vol = data["Strategy_Returns"].std()

# Annualized return
market_annual_return = (1 + market_daily_mean) ** 252 - 1
strategy_annual_return = (1 + strategy_daily_mean) ** 252 - 1

# Annualized volatility
market_annual_vol = market_daily_vol * np.sqrt(252)
strategy_annual_vol = strategy_daily_vol * np.sqrt(252)

# Sharpe ratio (risk-free rate assumed 0)
market_sharpe = market_annual_return / market_annual_vol
strategy_sharpe = strategy_annual_return / strategy_annual_vol

print("=== MARKET ===")
print(f"Annual Return: {market_annual_return:.4f}")
print(f"Annual Volatility: {market_annual_vol:.4f}")
print(f"Sharpe Ratio: {market_sharpe:.4f}")

print("\n=== STRATEGY ===")
print(f"Annual Return: {strategy_annual_return:.4f}")
print(f"Annual Volatility: {strategy_annual_vol:.4f}")
print(f"Sharpe Ratio: {strategy_sharpe:.4f}")

# ==============================
# 6. Max Drawdown
# ==============================

market_rolling_max = data["Market_Cumulative"].cummax()
market_drawdown = data["Market_Cumulative"] / market_rolling_max - 1
market_max_dd = market_drawdown.min()

strategy_rolling_max = data["Strategy_Cumulative"].cummax()
strategy_drawdown = data["Strategy_Cumulative"] / strategy_rolling_max - 1
strategy_max_dd = strategy_drawdown.min()

print(f"\nMarket Max Drawdown: {market_max_dd:.4f}")
print(f"Strategy Max Drawdown: {strategy_max_dd:.4f}")

# Return / Max Drawdown ratio
market_return_dd_ratio = market_annual_return / abs(market_max_dd)
strategy_return_dd_ratio = strategy_annual_return / abs(strategy_max_dd)

print(f"\nReturn / MaxDD Market: {market_return_dd_ratio:.4f}")
print(f"Return / MaxDD Strategy: {strategy_return_dd_ratio:.4f}")

# ==============================
# 7. Reusable Backtest Function
# ==============================

def backtest_sma(data, short_window, long_window):

    df = data.copy()

    short_window = max(1, int(short_window))
    long_window = max(1, int(long_window))

    df["SMA_short"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_long"] = df["Close"].rolling(window=long_window).mean()

    df["Signal"] = 0
    df.loc[df["SMA_short"] > df["SMA_long"], "Signal"] = 1

    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    mean_return = df["Strategy_Returns"].mean()
    vol = df["Strategy_Returns"].std()

    annual_return = (1 + mean_return) ** 252 - 1
    annual_vol = vol * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol != 0 else 0

    return annual_return, annual_vol, sharpe

# ==============================
# 8. Train / Test Split
# ==============================

train = data.loc["2020-01-01":"2022-12-31"]
test = data.loc["2023-01-01":"2023-12-31"]

# ==============================
# 9. Parameter Optimization (Grid Search)
# ==============================

results = []

short_range = range(5, 31, 5)
long_range = range(40, 201, 20)

for short_window in short_range:
    for long_window in long_range:

        if short_window < long_window:

            annual_return, annual_vol, sharpe = backtest_sma(
                data, short_window, long_window
            )

            results.append({
                "short_window": short_window,
                "long_window": long_window,
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": sharpe
            })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe", ascending=False)

print("\nTop 10 Parameter Combinations:")
print(results_df.head(10))


# ==============================
# 10. Optimization on TRAIN only
# ==============================

train_results = []

for short_window in short_range:
    for long_window in long_range:

        if short_window < long_window:

            annual_return, annual_vol, sharpe = backtest_sma(
                train, short_window, long_window
            )

            train_results.append({
                "short_window": short_window,
                "long_window": long_window,
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": sharpe
            })

train_results_df = pd.DataFrame(train_results)
train_results_df = train_results_df.sort_values(by="sharpe", ascending=False)

print("\nTop 5 TRAIN combinations:")
print(train_results_df.head())

# ==============================
# 10.1 Select Best Combination
# ==============================

best_short = train_results_df.iloc[0]["short_window"]
best_long = train_results_df.iloc[0]["long_window"]

print(f"\nBest parameters from TRAIN:")
print(f"Short Window: {best_short}")
print(f"Long Window: {best_long}")


# ==============================
# 11. Evaluation on TEST
# ==============================

test_return, test_vol, test_sharpe = backtest_sma(
    test, best_short, best_long
)

print("\n=== TEST PERFORMANCE (Out-of-Sample) ===")
print(f"Annual Return: {test_return:.4f}")
print(f"Annual Volatility: {test_vol:.4f}")
print(f"Sharpe Ratio: {test_sharpe:.4f}")


# ==============================
# 12. Buy & Hold Performance in TEST
# ==============================

# Daily stats (TEST only)
test_market_daily_mean = test["Returns"].mean()
test_market_daily_vol = test["Returns"].std()

# Annualized metrics
test_market_annual_return = (1 + test_market_daily_mean) ** 252 - 1
test_market_annual_vol = test_market_daily_vol * np.sqrt(252)

test_market_sharpe = test_market_annual_return / test_market_annual_vol

print("\n=== BUY & HOLD (TEST 2023) ===")
print(f"Annual Return: {test_market_annual_return:.4f}")
print(f"Annual Volatility: {test_market_annual_vol:.4f}")
print(f"Sharpe Ratio: {test_market_sharpe:.4f}")

# ==============================
# 12.1 Comparación directa
# - Buscamos:
#   Estrategia > Mercado = Generamos alpha real
#   Similar = Quizas no agregamos valor, pero reducimos riesgo tal vez
#   Mercado > Estrategia = La estrategia no es rentable o no agrega valor en este periodo
# ==============================

print("\n=== COMPARISON (TEST) ===")
print(f"Strategy Sharpe: {test_sharpe:.4f}")
print(f"Market Sharpe:   {test_market_sharpe:.4f}")
print(f"Sharpe Difference: {test_sharpe - test_market_sharpe:.4f}")