# IA

## Descripción

Aquí se tiene un codiggo de Python hecho  para ilustrar la implementación de un algoritmo de descenso de gradiente para entrenar un modelo de regresión lineal. El código utiliza el conjunto de datos "Iris" para realizar una regresión lineal simple. El objetivo es predecir la longitud del pétalo en función de las longitudes normalizadas del sépalo

## Cómo funciona el código

1. **Leer Datos y Crear DataFrame**: El código lee el archivo de datos y crea un DataFrame de Pandas con las columnas ("sepal length", "sepal width", "petal length", "petal width" y "class")

2. **Normalización de Datos**: Se realiza la normalización de las longitudes del sépalo dividiendo las longitudes por sus valores máximos respectivos

3. **Preparación de Datos**: Se extraen las características normalizadas de longitud del sépalo y se guardan en la variable "samples". La variable objetivo, que es la longitud del pétalo, se almacena en la variable "y". Se inicializan los parámetros para el modelo de regresión

4. **Definición de Funciones**: Se definen las funciones necesarias para el cálculo de la hipótesis, el descenso de gradiente y el cálculo del error medio cuadrado.

5. **Entrenamiento del Modelo**: Se realiza el entrenamiento del modelo utilizando el algoritmo de descenso de gradiente. Se itera sobre un número máximo de épocas y se ajustan los parámetros para minimizar el error medio cuadrado

6. **Finalización del Entrenamiento**: El entrenamiento del modelo se detiene si los parámetros convergen o se alcanza el número máximo de épocas. Los resultados finales, incluidos los parámetros del modelo, se muestran para un buen aanalisis

7. **Visualización de Errores**: Se crea una gráfica que muestra cómo disminuye el error medio cuadrado a lo largo de las épocas, SE PUEDE CONFIRMAR COMO ES QUE EL ESTE PROCESO SE REALIZA CORRECTAMENTE

## Conclusión
Aunque se observe como el error va disminuyendo conforme pasan las epocas, no es completamente bajo y no se acerca tanto a 1, esto se debe a la cantidad de datos que tenemos en nuestro dataset, pues estamos hablando de 153 datos.
