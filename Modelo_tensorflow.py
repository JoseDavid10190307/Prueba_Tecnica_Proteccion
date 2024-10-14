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
threshold_percentage = 0.20 # Por ejemplo, 20% por encima de la media
attractive_threshold = mean_price_per_sqft * (1 + threshold_percentage)

'''
print("Media del precio por sqft:", mean_price_per_sqft)
print("Desviación estándar del precio por sqft:", std_price_per_sqft)
print("Umbral de atractivo:", attractive_threshold)
'''

# Preprocesamiento
# Definir las características y la variable objetivo
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'price_per_sqft']]  # Ajusta según sea necesario
y = df['price_per_sqft'] * (1 + threshold_percentage)  # Aquí se calcula el porcentaje de atractivo

# Guardar los IDs originales
ids = df['id']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, ids, test_size=0.2, random_state=42)

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

# Crear un DataFrame con las predicciones
test_results = pd.DataFrame(X_test, columns=['sqft_living', 'bedrooms', 'bathrooms', 'price_per_sqft'])
test_results['attractiveness_percentage'] = predictions
test_results['id'] = ids_test.values  # Mantener los IDs originales

# Juntar con el DataFrame original para tener todas las columnas
df_test = pd.concat([df.set_index('id').loc[ids_test].reset_index(), test_results], axis=1)

# Filtrar las 10 propiedades más atractivas
top_attractive = df_test.nlargest(10, 'attractiveness_percentage')

# Crear el DataFrame "propiedades_Mejores_Invertir"
propiedades_Mejores_Invertir = top_attractive[['id', 'sqft_living', 'bedrooms', 'bathrooms', 'price', 'attractiveness_percentage']]
print("10 Propiedades Mejores para Invertir:")
print(propiedades_Mejores_Invertir)

'''
# Función para buscar una propiedad específica por su ID y devolver su porcentaje de atractivo
def buscar_propiedad_por_id(propiedad_id):
    propiedad = df_test[df_test['id'] == propiedad_id]
    if not propiedad.empty:
        return propiedad[['id', 'sqft_living', 'bedrooms', 'bathrooms', 'price', 'attractiveness_percentage']]
    else:
        return "Propiedad no encontrada en el conjunto de datos de prueba"

# Ejemplo de uso: buscar una propiedad específica por su ID
id_a_buscar = 855700170  # Cambia esto por el ID que quieras buscar
resultado = buscar_propiedad_por_id(id_a_buscar)
print(f"Información de la propiedad con ID {id_a_buscar}:")
print(resultado)

Esta ultima función presentó fallos con algunas propiedades en concreto, es por eso que esta comentada
'''
