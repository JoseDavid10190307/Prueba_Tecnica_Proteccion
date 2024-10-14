# Importar librerías necesarias para el análisis exploratorio y demas
import pandas as pd

# Cargar el conjunto de datos desde la URL proporcionada
url = "https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv"
df = pd.read_csv(url)

# Mostrar las primeras filas del dataset para entender su estructura
print(df.head())

# Verificar si hay valores nulos en el dataset
missing_values = df.isnull().sum()

# Estadísticas descriptivas del conjunto de datos
statistics = df.describe()

# Filtrar solo las columnas numéricas para calcular la matriz de correlación
numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación solo con las columnas numéricas
correlation_matrix = numeric_columns.corr()

# Mostrar los valores de correlación con respecto al precio
correlation_with_price = correlation_matrix['price'].sort_values(ascending=False)

# Mostrar resultados
print("\nValores nulos:\n", missing_values)
print("\nEstadísticas descriptivas:\n", statistics)
print("\nMatriz de correlación:\n", correlation_matrix)
print("\nCorrelación con el precio:\n", correlation_with_price)


'''
Generación de los Gráficos de la imagen

import matplotlib.pyplot as plt
import seaborn as sns

# Ajustar el tamaño de los gráficos
plt.figure(figsize=(12, 8))

# Histograma del precio de las viviendas
plt.subplot(2, 2, 1)
sns.histplot(df['price'], bins=50, kde=True, color='blue')
plt.title('Distribución del Precio de las Viviendas')

# Relación entre sqft_living y precio
plt.subplot(2, 2, 2)
sns.scatterplot(x='sqft_living', y='price', data=df, alpha=0.5)
plt.title('Relación entre Superficie Habitable y Precio')

# Relación entre grade y precio
plt.subplot(2, 2, 3)
sns.boxplot(x='grade', y='price', data=df)
plt.title('Distribución de Precios por Calificación (Grade)')

# Relación entre waterfront y precio
plt.subplot(2, 2, 4)
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('Relación entre Propiedades Frente al Agua y Precio')

plt.tight_layout()
plt.show()

'''

