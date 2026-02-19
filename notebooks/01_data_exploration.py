import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Elegimos acción
ticker = "AAPL"

# Descargamos datos
data = yf.download(
    ticker, 
    start="2020-01-01", 
    end="2024-01-01", 
    auto_adjust=True) # Ajusta los precios para dividencias y splits

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


# Mostramos la gráfica
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Precio de Cierre')
plt.title(f'Precio de Cierre de {ticker}')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.grid(True)
plt.show()