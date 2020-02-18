"""
@author: Karl Behrens
@email: karlbehrensg@gmail.com
"""

# Regresion Lineal Polinomica

# Como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Ajustar la regresion polinomica con el dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualizacion de los resultados de los modelos
plt.scatter(X, y, c = 'red')
plt.plot(X, lin_reg.predict(X), c = 'blue')
plt.plot(X, lin_reg_2.predict(X_poly), c = 'green')
plt.title("Rendimientos de modelos")
plt.ylabel("Sueldo (en $")
plt.xlabel(("Posicion del empleado"))
plt.legend(["Lineal","Polinomico"])
plt.show()


