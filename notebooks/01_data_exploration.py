import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# Elegimos acción
ticker = "AAPL"

# Descargamos datos
data = yf.download(
    ticker, 
    start="2020-01-01", 
    end="2024-01-01", 
    auto_adjust=True
)

# Si las columnas son multiIndex, las simplificamos
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

    print("Columnas simplificadas:", data.columns)

# Retorno diario
data['Retorno'] = data['Close'].pct_change()
print("\nRetorno diario:")
print(data['Retorno'].describe())

# Mostramos los primeros registros
print("\nMostramos los 5 primeros registros:")
print("-----------------------------------")
print(data.head())
print("-----------------------------------")
print(data.info())

print("\nValores nulos por columna:")
print("-----------------------------------")
print(data.isnull().sum())
print("-----------------------------------")

# Medias móviles
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

# Creamos la señal
data["Signal"] = 0
data.loc[data["SMA_20"] > data["SMA_50"], "Signal"] = 1

# Estrategia de retorno
data["Retorno_Estrategia"] = data["Signal"].shift(1) * data["Retorno"]

# Rendimiento acumulado
data["Acumulado_Mercado"] = (1 + data["Retorno"]).cumprod() 
data["Acumulado_Estrategia"] = (1 + data["Retorno_Estrategia"]).cumprod()

# Calcular: 
# - Retorno anualizado
# - Volatilidad anualizada
# - Sharpe Ratio

# Retornos promedios diarios
retorno_diario_mercado = data["Retorno"].mean()
retorno_diario_estrategia = data["Retorno_Estrategia"].mean()

# Volatilidad promedio diaria
volatilidad_diaria_mercado = data["Retorno"].std()
volatilidad_diaria_estrategia = data["Retorno_Estrategia"].std()

# Retornos anualizados
retorno_anualizado_mercado = (1 + retorno_diario_mercado) ** 252 - 1
retorno_anualizado_estrategia = (1 + retorno_diario_estrategia) ** 252 - 1

# Volatilidad anualizada
volatilidad_anualizada_mercado = volatilidad_diaria_mercado * np.sqrt(252)
volatilidad_anualizada_estrategia = volatilidad_diaria_estrategia * np.sqrt(252)

# Sharpe Ratio
sharpe_ratio_mercado = retorno_anualizado_mercado / volatilidad_anualizada_mercado
sharpe_ratio_estrategia = retorno_anualizado_estrategia / volatilidad_anualizada_estrategia

print("=== MARKET ===")
print(f"Retorno anualizado promedio mercado: {retorno_anualizado_mercado:.4f}")
print(f"Volatilidad anualizada promedio mercado: {volatilidad_anualizada_mercado:.4f}")
print(f"Sharpe Ratio promedio mercado: {sharpe_ratio_mercado:.4f}")

print("\n=== STRATEGY ===")
print(f"Retorno anualizado promedio estrategia: {retorno_anualizado_estrategia:.4f}")
print(f"Volatilidad anualizada promedio estrategia: {volatilidad_anualizada_estrategia:.4f}")
print(f"Sharpe Ratio promedio estrategia: {sharpe_ratio_estrategia:.4f}")


# Mercado drawdown
rolling_max_mercado = data["Acumulado_Mercado"].cummax()
drawdown_mercado = data["Acumulado_Mercado"] / rolling_max_mercado - 1
max_drawdown_mercado = drawdown_mercado.min()

# Estrategia drawdown
rolling_max_estrategia = data["Acumulado_Estrategia"].cummax()
drawdown_estrategia = data["Acumulado_Estrategia"] / rolling_max_estrategia - 1
max_drawdown_estrategia = drawdown_estrategia.min()

print(f"\nMax Drawdown Mercado: {max_drawdown_mercado:.4f}")
print(f"\nMax Drawdown Estrategia: {max_drawdown_estrategia:.4f}")


# Retorno ratio / Max Drawdown
ratio_mercado = retorno_anualizado_mercado / abs(max_drawdown_mercado)
ratio_estrategia = retorno_anualizado_estrategia / abs(max_drawdown_estrategia)

print(f"\nRetorno / MaxDD Mercado: {ratio_mercado:.4f}")
print(f"\nRetorno / MaxDD Estrategia: {ratio_estrategia:.4f}")



# Comparación visual
plt.figure(figsize=(10, 6))
plt.plot(data["Acumulado_Mercado"], label="Buy and Hold")
plt.plot(data["Acumulado_Estrategia"], label="SMA Estrategia")
plt.title(f"Comparación de Rendimientos: {ticker}")
plt.xlabel("Fecha")
plt.ylabel("Rendimiento Acumulado")
plt.grid(True)
plt.show()


# Mostramos la gráfica
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Precio de Cierre')
plt.title(f'Precio de Cierre de {ticker}')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.grid(True)
plt.show()