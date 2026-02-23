import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# ==============================
# 1. Descargar Data
# ==============================

ticker = "MSFT"

data = yf.download(
    ticker,
    start="2020-01-01",
    end="2024-01-01",
    auto_adjust=True
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data["Returns"] = data["Close"].pct_change()

# ==============================
# 2. Train / Test Split
# ==============================

train = data.loc["2020-01-01":"2021-12-31"].copy()
test = data.loc["2022-01-01":"2022-12-31"].copy()

# ==============================
# 3. Backtest EMA + Momentum
# ==============================

def backtest_ema_momentum(data, short_window, long_window, momentum_window):

    df = data.copy()

    df["EMA_short"] = df["Close"].ewm(span=short_window, adjust=False).mean()
    df["EMA_long"] = df["Close"].ewm(span=long_window, adjust=False).mean()
    df["Momentum"] = df["Close"].pct_change(momentum_window)

    df["Signal"] = 0
    df.loc[
        (df["EMA_short"] > df["EMA_long"]) &
        (df["Momentum"] > 0),
        "Signal"
    ] = 1

    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    df["Market_Cumulative"] = (1 + df["Returns"]).cumprod()
    df["Strategy_Cumulative"] = (1 + df["Strategy_Returns"]).cumprod()

    # Annual metrics
    mean_return = df["Strategy_Returns"].mean()
    vol = df["Strategy_Returns"].std()

    annual_return = (1 + mean_return) ** 252 - 1
    annual_vol = vol * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0

    # Max Drawdown
    roll_max = df["Strategy_Cumulative"].cummax()
    drawdown = df["Strategy_Cumulative"] / roll_max - 1
    max_dd = drawdown.min()

    return annual_return, annual_vol, sharpe, max_dd, df

# ==============================
# 4. Optimización SOLO en TRAIN
# ==============================

short_range = range(5, 31, 5)
long_range = range(40, 201, 20)
momentum_range = [30, 60, 90, 120]

results = []

for short_window in short_range:
    for long_window in long_range:
        for momentum_window in momentum_range:

            if short_window < long_window:

                annual_return, annual_vol, sharpe, max_dd, _ = backtest_ema_momentum(
                    train,
                    short_window,
                    long_window,
                    momentum_window
                )

                results.append({
                    "short": short_window,
                    "long": long_window,
                    "momentum": momentum_window,
                    "return": annual_return,
                    "vol": annual_vol,
                    "sharpe": sharpe,
                    "max_dd": max_dd
                })

results_df = pd.DataFrame(results).sort_values(by="sharpe", ascending=False)

print("\nTop 5 TRAIN combinations (EMA + Momentum):")
print(results_df.head())

# ==============================
# 5. Seleccionar Mejor Parámetro
# ==============================

best = results_df.iloc[0]

best_short = int(best["short"])
best_long = int(best["long"])
best_momentum = int(best["momentum"])

print("\nBest Parameters from TRAIN:")
print(f"Short: {best_short}")
print(f"Long: {best_long}")
print(f"Momentum: {best_momentum}")

# ==============================
# 6. Evaluación en TEST
# ==============================

test_return, test_vol, test_sharpe, test_dd, test_df = backtest_ema_momentum(
    test,
    best_short,
    best_long,
    best_momentum
)

print("\n=== TEST PERFORMANCE (2022) ===")
print(f"Strategy Return: {test_return:.4f}")
print(f"Strategy Volatility: {test_vol:.4f}")
print(f"Strategy Sharpe: {test_sharpe:.4f}")
print(f"Strategy MaxDD: {test_dd:.4f}")

# ==============================
# 7. Buy & Hold en TEST
# ==============================

mean_market = test["Returns"].mean()
vol_market = test["Returns"].std()

market_return = (1 + mean_market) ** 252 - 1
market_vol = vol_market * np.sqrt(252)
market_sharpe = market_return / market_vol

market_cum = (1 + test["Returns"]).cumprod()
market_roll = market_cum.cummax()
market_dd = (market_cum / market_roll - 1).min()

print("\n=== BUY & HOLD (2022) ===")
print(f"Market Return: {market_return:.4f}")
print(f"Market Volatility: {market_vol:.4f}")
print(f"Market Sharpe: {market_sharpe:.4f}")
print(f"Market MaxDD: {market_dd:.4f}")

print("\n=== COMPARISON ===")
print(f"Sharpe Difference: {test_sharpe - market_sharpe:.4f}")
print(f"Return Difference: {test_return - market_return:.4f}")
print(f"MaxDD Difference: {test_dd - market_dd:.4f}")

# ==============================
# 8. Gráfico TEST
# ==============================

plt.figure(figsize=(10,6))
plt.plot(test_df["Market_Cumulative"], label="Buy & Hold")
plt.plot(test_df["Strategy_Cumulative"], label="EMA + Momentum")
plt.title(f"{ticker} - TEST 2022")
plt.legend()
plt.grid(True)
plt.show()