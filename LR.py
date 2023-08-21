import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------

# Obtener el directorio actual (carpeta en la que nos encontramos)
current_directory = os.getcwd()
print("Directorio actual:", current_directory)
# Concatenar la ruta del archivo de datos a la ruta del directorio actual
data_file_path = os.path.join(current_directory, "iris.data")
print("Ruta completa del archivo:", data_file_path)

#---------------------------------------------------------------------------------------------

# Leer el archivo de datos .data y almacenar los datos en un DataFrame
columns = ["sepal length","sepal width","petal length","petal width", "class"]
df = pd.read_csv(data_file_path, names=columns)
print(df.head())

#---------------------------------------------------------------------------------------------

# Normalizar los datos
df["sepal lengthN"] = df["sepal length"] / df["sepal length"].max()
df["sepal widthN"] = df["sepal width"] / df["sepal width"].max()

#---------------------------------------------------------------------------------------------

samples = df[["sepal lengthN", "sepal widthN"]].to_numpy().tolist() #Estas son las características
y = df['petal length'].to_numpy().tolist() #Esta es la variable a predecir
params = np.zeros(len(samples[0])+ 1).tolist() #Se especifica N+1 porque se agrega el término independiente
print("SAMPLES:", len(samples[0]))

#---------------------------------------------------------------------------------------------

"""
Podriamos usar una funcion que divida los datos en conjuntos de entrenamiento y prueba, 
pero en este caso se hará manualmente.
Recordemos que si no se dividen los datos, el modelo se ajusta perfectamente a los datos de entrenamiento,
y por lo tanto, no se puede evaluar su desempeño para datos nuevos. Entonces, tendriamos lo que se llama
un modelo sobreajustado (overfitting).
"""

#---------------------------------------------------------------------------------------------

# Configurar el learning rate y el número máximo de épocas
alfa = 0.01
max_epochs = 1000
__errors__ = []

#---------------------------------------------------------------------------------------------

# Función para calcular la hipótesis
def hyp(params, sample):
    acum = 0
    for p, x in zip(params, sample):
        acum = acum + p * x
    return acum

#---------------------------------------------------------------------------------------------

# Función para realizar el descenso de gradiente
def gd(params, samples, y, alfa):
    temp = []
    for i in range(len(params)):
        acum = 0
        for j in range(len(samples)):
            error = (hyp(params, samples[j]) - y[j])
            acum = acum + error * samples[j][0]
        temp.append(params[i] - alfa / len(samples) * acum)
    return temp

#---------------------------------------------------------------------------------------------

# Función para calcular el error
def errors(params, samples, y):
    global __errors__
    acum = 0
    for i in range(len(samples)):
        h = hyp(params, samples[i])
        print( 'hyp: %f  y: %f'  % (h, y[i]))
        error = (h - y[i]) ** 2
        acum = acum + error
    mean_error = acum / len(samples)
    __errors__.append(mean_error)
    return mean_error

#---------------------------------------------------------------------------------------------

# Entrenamiento del modelo
for epoch in range(max_epochs):
    oldparams = params.copy()
    params = gd(params, samples, y, alfa)
    error = errors(params, samples, y)
    
    #print(f"Epoch {epoch + 1}: Error = {error:.4f}")
    
    """
    np.allclose(oldparams, params) compara los valores de los parámetros de la época anterior
    El punto es checar si los parametros son relativamente cercanos a los de la época anterior
    Si son cercanos, entonces ya no hay necesidad de seguir entrenando, y se termina el ciclo.
    
    """
    if np.allclose(oldparams, params) or epoch == max_epochs - 1:
        print("END!!!!!!!!!")
        break

#---------------------------------------------------------------------------------------------

# Imprimir los samples y los final params
print("Samples:")
for s in samples:
    print(s)
print("Final parameters:")
print(params)
print("final error:")
#print(__errors__)
#---------------------------------------------------------------------------------------------

# Graficar los errores
plt.plot(__errors__)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Error vs. Epoch')
plt.show()

#---------------------------------------------------------------------------------------------