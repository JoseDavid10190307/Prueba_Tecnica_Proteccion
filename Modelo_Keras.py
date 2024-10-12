# Importar librerías necesarias para el modelo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Cargar el conjunto de datos desde la URL proporcionada
url = "https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv"
df = pd.read_csv(url)

# Calcular el valor por metro cuadrado (precio por sqft)
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Calcular la media y la desviación estándar del precio por sqft
mean_price_per_sqft = df['price_per_sqft'].mean()
std_price_per_sqft = df['price_per_sqft'].std()

# Establecer un porcentaje para determinar el umbral de atractivo
# Por ejemplo, 20% por encima de la media
threshold_percentage = 0.20
attractive_threshold = mean_price_per_sqft * (1 + threshold_percentage)

print("Media del precio por sqft:", mean_price_per_sqft)
print("Desviación estándar del precio por sqft:", std_price_per_sqft)
print("Umbral de atractivo:", attractive_threshold)

# Preprocesamiento
# Definir las características y la variable objetivo
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'price_per_sqft']]  # Ajusta según sea necesario
y = df['price_per_sqft'] * (1 + threshold_percentage)  # Aquí se calcula el porcentaje de atractivo

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construcción del modelo usando TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # Salida continua para el porcentaje de atractivo
])

# Compilación del modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluación del modelo
loss = model.evaluate(X_test, y_test)
print("Pérdida en el conjunto de prueba:", loss)

# Predicción del porcentaje de atractivo
predictions = model.predict(X_test)

# Agregar las predicciones al DataFrame
test_results = pd.DataFrame(X_test, columns=['sqft_living', 'bedrooms', 'bathrooms', 'price_per_sqft'])
test_results['attractiveness_percentage'] = predictions

# Juntar las propiedades más atractivas para invertir
test_results['index'] = test_results.index  # Agregar un índice para combinar después
df_test = pd.concat([df.reset_index(drop=True), test_results.reset_index(drop=True)], axis=1)

# Filtrar las 10 propiedades más atractivas
top_attractive = df_test.nlargest(10, 'attractiveness_percentage')

# Crear el DataFrame "10propiedades_Mejores_Invertir"
propiedades_mejores_invertir = top_attractive[['index', 'sqft_living', 'bedrooms', 'bathrooms', 'price', 'attractiveness_percentage']]
print(propiedades_mejores_invertir)

# Imprimir el DataFrame de las 10 mejores propiedades para invertir
print("10 Propiedades Mejores para Invertir:")
print(propiedades_mejores_invertir)
