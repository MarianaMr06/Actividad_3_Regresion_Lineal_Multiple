#Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#Carga de Archivos
credicel = pd.read_csv ("Credicel_Limpio_Final.csv")
credicel.info()
#Enganche
    # semana
    # monto_financiado
    # Costo_total
    # monto_accesorios
    # Puntos
    # score_buro

#Declaración de variables
x_para_enganche = credicel[["semana", "monto_financiado", "costo_total", "monto_accesorios", "puntos", "score_buro"]]
y_enganche = credicel["enganche"]

#Modelo
model = LinearRegression()
type(model)

model.fit(X = x_para_enganche, y = y_enganche)
model.__dict__

#Coeficiente de determinación
determinacion_enganche = model.score(x_para_enganche, y_enganche)
correlacion_enganche = np.sqrt(determinacion_enganche)