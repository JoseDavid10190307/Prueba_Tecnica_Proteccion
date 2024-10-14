# Documentación del Proyecto: Análisis de Datos de Precios de Viviendas

## 1. Introducción
Este proyecto tiene como objetivo predecir el **atractivo de propiedades inmobiliarias** basándose en sus características y definir cuáles son las **10 mejores propiedades para invertir**. Utilizamos un modelo de regresión con la biblioteca TensorFlow para calcular el porcentaje de atractivo de cada propiedad, comparándolo con el promedio del precio por metro cuadrado. El proyecto incluye análisis exploratorio, preprocesamiento de datos, entrenamiento de un modelo, predicciones y la generación de un DataFrame con las mejores propiedades para invertir.

## 2. Preprocesamiento de Datos
- **Dataset utilizado:** El dataset se descargó de [aquí](https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv). Contiene características de propiedades como tamaño en pies cuadrados (`sqft_living`), número de habitaciones, baños y el precio de venta.
- **Ingeniería de características:** Se calculó el precio por pie cuadrado (`price_per_sqft`) dividiendo el precio de la propiedad por su tamaño en pies cuadrados. Este valor se utilizó como base para el cálculo del atractivo de la propiedad.
- **Definición del umbral de atractivo:** El atractivo de las propiedades se definió como un **20% por encima** del promedio de `price_per_sqft`. Las propiedades que superan este umbral se consideran atractivas.
- **Características seleccionadas:** Las características seleccionadas para el modelo son `sqft_living`, `bedrooms`, `bathrooms` y `price_per_sqft`, ya que son representativas del valor de una propiedad.

## 3. División de los Datos
- **División del dataset:** Usamos `train_test_split` de `sklearn` para dividir el dataset en **80% de entrenamiento** y **20% de prueba**, manteniendo los IDs de las propiedades para facilitar la validación.
- **Escalado de características:** Las características fueron escaladas utilizando `StandardScaler` para mejorar el rendimiento del modelo.

## 4. Entrenamiento del Modelo
- **Modelo utilizado:** Se implementó un modelo de regresión con **TensorFlow/Keras**. El modelo tiene dos capas densas ocultas con 64 y 32 unidades, y una capa de salida para predecir el porcentaje de atractivo.
- **Función de pérdida y optimizador:** Se utilizó el optimizador **Adam** y la función de pérdida fue el **mean_squared_error**, una elección estándar para problemas de regresión.
- **Entrenamiento:** El modelo fue entrenado durante **100 épocas** con un tamaño de batch de 32.

## 5. Predicción y Evaluación
- **Evaluación:** El modelo se evaluó en el conjunto de prueba, utilizando el error cuadrático medio (**MSE**) para medir el rendimiento.
- **Predicción de atractivo:** Se predijo el **porcentaje de atractivo** de las propiedades en el conjunto de prueba, comparando estos valores con el umbral definido.

## 6. Selección de las Mejores Propiedades
- **Selección de las 10 mejores propiedades:** Se creó un DataFrame llamado `propiedades_Mejores_Invertir`, que contiene las **10 propiedades con mayor porcentaje de atractivo**. Este DataFrame incluye las características clave y el ID original de cada propiedad.
- **Validación de resultados:** Se comprobó que las 10 propiedades más atractivas son reales y superan el promedio esperado, validando así las predicciones.

## 7. Funcionalidad de Búsqueda
- Se implementó una función que permite **buscar una propiedad por su ID** y devolver su información y porcentaje de atractivo, facilitando la validación de propiedades particulares (aunque esta funcionalidad no se completó al 100%).

## 8. Visualización de Resultados
- **Visualización de relaciones:** Se generaron gráficos que muestran:
  - La relación entre **superficie habitable** y **precio**.
  - La distribución de los precios iniciales del dataset.
  - La relación entre las propiedades frente al agua y su precio.
  - La distribución de precios según la calificación de las propiedades.

## 9. Conclusión y Futuras Mejoras
- **Conclusión:** El modelo logró identificar propiedades atractivas basadas en sus características. Las **10 mejores propiedades** mostraron un porcentaje de atractivo significativamente mayor al umbral definido, validando así el enfoque del proyecto.
- **Futuras mejoras:** Incluir un análisis más profundo de las características influyentes, agregar variables como **ubicación** o **año de construcción**, y probar con algoritmos más avanzados como **Random Forest** o **Gradient Boosting**. Además, finalizar la funcionalidad de búsqueda.

## 10. Referencias
- **Datos descargados de:** [home_data.csv](https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv)
- **Biblioteca de Machine Learning:** TensorFlow/Keras
- **Preprocesamiento y Modelado:** Scikit-learn, Pandas, Numpy
