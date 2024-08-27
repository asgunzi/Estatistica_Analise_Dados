#Regressão Logística
#Copie este código, cole na sua IDE do Python favorita e execute para ver os resultados

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

from sklearn.linear_model import LogisticRegression


dados = pd.DataFrame({'Prova_Logica': [2, 2, 5, 5, 5, 2, 3, 2, 1, 4, 
                 5, 8, 1, 1, 3, 4, 3, 2, 1, 1, 8, 8, 1, 2, 1, 5, 3, 3, 5, 4, 4, 
                 1, 8, 3, 2, 3, 3, 2, 1, 1, 5, 4, 1, 5, 3, 1, 4, 6, 1, 1, 8, 1, 
                 1, 5, 1, 5, 3, 1, 1, 8, 1, 1, 1, 1, 1, 2, 1, 5, 5, 4, 2, 1, 8, 
                 4, 5, 1, 3, 3, 3, 5, 3, 1, 7, 1, 1, 2, 9, 5, 3, 1, 5, 1, 4, 2, 
                 1, 4, 3, 3, 8, 1, 1, 8, 5, 1, 1, 1, 5, 8, 5, 1, 4, 2, 5, 4, 5, 
                 3, 3, 5, 5, 5, 5, 5, 8, 5, 4, 9, 8, 1, 3, 4, 2, 5, 1, 4, 3, 5, 
                 5, 5, 6, 4, 3, 5, 7, 1, 8, 5, 7, 3, 2, 3, 2, 5, 5, 5, 5, 4, 4, 
                 8, 1, 1, 2, 5, 3, 2, 7, 4, 1, 1, 1, 4, 5, 1, 1, 8, 3, 6, 8, 3, 
                 1, 3, 3, 2, 8, 4, 1, 1, 1, 1, 1, 2, 3, 4, 6, 2, 3, 3, 4, 2, 1, 
                 5, 2, 4, 3, 3, 1, 3, 3, 3, 1, 3, 5, 6, 1, 5, 1, 5, 4, 3, 1, 6, 
                 1, 4, 9, 3, 3, 2, 1, 1, 4, 3, 1, 3, 1, 1, 3, 7, 8, 1, 3, 5, 6, 
                 3, 6, 5, 8, 5, 1, 1, 4, 2, 1, 8, 7, 5, 1, 1, 1, 6, 5, 7, 3, 3, 
                 5, 1, 3, 5, 1, 8, 8, 1, 2, 3, 3, 3, 3, 7, 1, 9, 8, 4, 1, 7, 1, 
                 1, 1, 5, 1, 1, 5, 3, 5, 1, 3, 6, 2, 1, 3, 4, 5, 6, 1, 5, 1, 5, 
                 1, 1, 5, 1, 1, 1, 5, 1, 3, 7, 1, 4, 3, 7, 1, 1, 5, 4, 1, 1, 3, 
                 5, 4, 2, 1, 5, 1, 1, 1, 8, 8, 5, 1, 2, 1, 6, 8, 3, 1, 5, 1, 5, 
                 1, 4, 4, 8, 1, 1, 1, 5, 1, 1, 5, 4, 6, 8, 1, 3, 1, 6, 1, 1, 1, 
                 1, 1, 8, 1, 5, 3, 1, 4, 4, 7, 2, 3, 3, 5, 8, 3, 1, 4, 1, 5, 1, 
                 7, 2, 6, 4, 1, 3, 1, 8, 5, 5, 5, 3, 1, 4, 5, 3, 6, 1, 3, 3, 5, 
                 4, 5, 3, 1, 4, 5, 1, 3, 6, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 5, 4, 
                 8, 1, 5, 6, 6, 4, 2, 5, 5, 6, 1, 2, 5, 6, 4, 2, 1, 2, 7, 2, 8, 
                 1, 3, 1, 1, 1, 6, 1, 4, 3, 1, 5, 2, 1, 1, 5, 4, 3, 1, 3, 1, 1, 
                 2, 4, 5, 4, 5, 4, 3, 7, 4, 1, 7, 1, 8, 5, 3, 3, 5, 1, 2, 1, 5, 
                 4, 4, 4, 3, 6, 8, 8, 1, 1, 1, 8, 8, 1, 5, 1, 4, 5, 1, 4, 1, 4, 
                 1, 5, 1, 4, 4, 1, 1, 2, 5, 1, 4, 4, 5, 4, 1, 1, 5, 3, 5, 4, 1, 
                 1, 5, 3, 3, 1, 5, 5, 4, 5, 7, 2, 5, 9, 4, 6, 1, 1, 1, 1, 8, 7, 
                 2, 3, 2, 9, 3, 1, 5, 1, 3, 5, 1, 4, 6, 3, 1, 5, 1, 3, 9, 1, 4, 
                 8, 8, 1, 3, 2, 3, 3, 1, 5, 1, 1, 3, 1, 5, 3, 4, 3, 4, 1, 4, 3, 
                 5, 1, 5, 2, 5, 8, 9, 4, 7, 4, 8, 5, 7, 5, 3, 5, 6, 3, 1, 5, 6, 
                 4, 3, 5, 1, 2, 4, 1, 1, 2, 9, 5, 1, 5, 1, 2, 1, 5, 1, 2, 3, 5, 
                 1, 1, 5, 3, 9, 5, 6, 9, 6, 1, 1, 9, 1, 3, 5, 4, 8, 4, 2, 7, 1, 
                 6, 3, 7, 8, 1, 5, 5, 6, 1, 4, 4, 3, 1, 5, 5, 8, 3, 3, 1, 9, 5, 
                 5, 5, 5, 1, 9, 6, 3, 1, 1, 2, 4, 6, 5, 7, 6, 5, 1], 'Redacao': [1, 
                 1, 1, 4, 3, 3, 5, 5, 9, 1, 1, 1, 1, 1, 4, 3, 3, 1, 1, 1, 1, 7, 
                 1, 1, 8, 1, 1, 1, 3, 8, 3, 1, 5, 3, 3, 1, 2, 7, 1, 1, 1, 1, 1, 
                 1, 7, 1, 1, 8, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 8, 5, 
                 1, 1, 1, 1, 1, 2, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 8, 5, 1, 
                 5, 8, 1, 1, 1, 5, 1, 1, 1, 2, 3, 3, 1, 3, 8, 1, 4, 6, 1, 1, 1, 
                 1, 3, 1, 1, 2, 3, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1, 8, 1, 5, 
                 1, 1, 1, 1, 1, 3, 6, 1, 2, 5, 6, 1, 2, 2, 1, 8, 1, 4, 6, 9, 3, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 
                 1, 3, 5, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1, 2, 1, 3, 1, 1, 1, 4, 5, 
                 1, 1, 6, 1, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 
                 6, 1, 8, 1, 1, 5, 1, 8, 2, 6, 1, 5, 1, 6, 1, 1, 1, 1, 1, 1, 1, 
1, 3, 1, 4, 8, 1, 1, 1, 8, 1, 3, 1, 6, 3, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 5, 1, 1, 3, 1, 1, 7, 4, 1, 1, 1, 1, 6, 1, 3, 
1, 4, 1, 1, 7, 2, 6, 4, 1, 1, 1, 1, 1, 4, 7, 1, 3, 1, 1, 9, 1, 
1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 2, 1, 1, 1, 5, 3, 1, 
1, 2, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 7, 1, 
1, 3, 8, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 2, 7, 1, 3, 1, 1, 1, 4, 
2, 4, 2, 2, 5, 3, 1, 1, 1, 5, 1, 9, 1, 1, 3, 2, 1, 1, 5, 1, 2, 
1, 3, 8, 1, 5, 1, 4, 3, 1, 8, 1, 6, 5, 1, 1, 1, 1, 1, 4, 5, 1, 
7, 8, 1, 4, 1, 1, 1, 1, 4, 1, 1, 2, 1, 8, 2, 6, 2, 1, 4, 1, 1, 
1, 1, 1, 4, 1, 1, 1, 1, 1, 8, 1, 1, 1, 3, 1, 1, 1, 8, 1, 1, 1, 
3, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 
3, 1, 2, 7, 2, 1, 1, 1, 1, 1, 2, 2, 1, 3, 1, 1, 3, 1, 1, 5, 1, 
7, 1, 1, 1, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 6, 8, 8, 7, 2, 
1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 9, 1, 8, 1, 1, 2, 
4, 1, 1, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 6, 1, 2, 
1, 1, 5, 4, 1, 8, 4, 6, 6, 1, 1, 1, 9, 1, 1, 1, 1, 1, 8, 1, 1, 
1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 4, 1, 1, 3, 2, 3, 1, 2, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 4, 1, 4, 2, 1, 6, 1, 
4, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 6, 1, 1, 1, 3, 2, 8, 1, 1, 
1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 6, 1, 6, 7, 1, 1, 5, 1, 2, 5, 
1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 8, 7, 1, 1, 1, 1, 4, 1, 6, 1, 
2, 8, 4, 7, 1, 1, 1, 5, 1, 1, 2, 1, 1, 7, 1, 1, 1, 4, 1, 1, 3, 
1, 5, 1, 7, 1], 'Auto_Avaliacao' : [1, 1, 1, 1, 1, 1, 1, 7, 1, 
    1, 1, 9, 1, 1, 1, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 3, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 8, 1, 1, 9, 7, 1, 2, 
    1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 3, 8, 1, 1, 1, 1, 1, 1, 6, 1, 1, 
    1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 8, 3, 1, 6, 1, 6, 1, 1, 3, 1, 1, 
    1, 1, 8, 5, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 6, 1, 2, 1, 1, 1, 1, 
    1, 4, 1, 6, 1, 1, 1, 2, 3, 1, 4, 7, 6, 1, 1, 1, 1, 1, 1, 9, 1, 
    2, 1, 1, 1, 1, 2, 1, 8, 1, 8, 1, 3, 2, 1, 1, 1, 1, 2, 6, 1, 1, 
    1, 2, 1, 3, 1, 1, 8, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 
    1, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 3, 5, 1, 1, 1, 1, 1, 1, 3, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 6, 1, 8, 1, 2, 5, 2, 1, 
    1, 7, 1, 1, 1, 8, 1, 1, 5, 1, 1, 1, 3, 1, 1, 1, 9, 1, 1, 1, 5, 
    9, 1, 5, 3, 4, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 3, 3, 1, 3, 1, 
    1, 1, 1, 1, 1, 3, 8, 1, 4, 1, 4, 1, 1, 1, 6, 1, 3, 1, 1, 2, 3, 
    1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 2, 2, 8, 1, 1, 1, 1, 1, 4, 1, 
    1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 4, 3, 1, 1, 4, 1, 6, 2, 1, 5, 3, 
    1, 1, 2, 1, 1, 1, 1, 1, 1, 8, 3, 4, 8, 1, 3, 7, 7, 1, 1, 2, 1, 
    1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 9, 1, 1, 1, 
    1, 8, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 9, 6, 3, 3, 
    1, 2, 1, 8, 7, 1, 1, 1, 1, 1, 6, 3, 1, 4, 5, 1, 6, 1, 1, 1, 1, 
    3, 1, 1, 2, 1, 6, 3, 7, 1, 8, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 8, 1, 1, 1, 9, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 2, 
    1, 1, 1, 2, 1, 7, 1, 1, 4, 2, 1, 5, 1, 5, 1, 1, 1, 4, 6, 1, 1, 
    4, 1, 2, 1, 1, 1, 9, 8, 1, 9, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 
    1, 1, 4, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 3, 1, 1, 8, 1, 1, 6, 1, 
    1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 5, 1, 6, 3, 1, 4, 3, 1, 7, 5, 1, 
    1, 1, 1, 1, 1, 2, 1, 5, 9, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 
    1, 2, 5, 1, 1, 5, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 5, 1, 8, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 6, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 
    1, 1, 1, 6, 1, 9, 5, 3, 1, 8, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 
    1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 2, 8, 1, 1, 1, 6, 1, 1, 1, 1, 9, 1, 1, 1],
'Classe': ["Ruim", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Boa", "Boa", "Ruim", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", 
      "Boa", "Boa", "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Boa", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Boa", "Ruim", 
      "Boa", "Boa", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Boa", "Boa", "Boa", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Boa", "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", 
      "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", "Boa", "Boa", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Boa", "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Boa", "Boa", "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Boa", "Boa", "Boa", 
      "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Boa", "Boa", "Ruim", "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Boa", "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Boa", "Ruim", "Boa", "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", 
      "Boa", "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Boa", "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Boa", "Boa", "Ruim", "Boa", "Boa", "Ruim", "Boa", "Boa", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Boa", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", 
      "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", 
      "Boa", "Boa", "Boa", "Boa", "Boa", "Boa", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Ruim", "Boa", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", 
      "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", 
      "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Boa", "Ruim", 
      "Boa", "Boa", "Ruim", "Boa", "Boa", "Boa", "Boa", "Ruim", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Boa", "Ruim", 
      "Ruim", "Boa", "Ruim", "Boa", "Boa", "Boa", "Ruim", "Boa", "Ruim", 
      "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Boa", "Boa", "Boa", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", 
      "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Boa", "Ruim", "Boa", "Boa", "Boa", "Boa", "Boa", "Boa", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", 
      "Ruim", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Boa", "Ruim", 
      "Ruim", "Ruim", "Boa", "Boa", "Ruim", "Boa", "Ruim", "Ruim", 
      "Boa", "Boa", "Boa", "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Ruim", 
      "Ruim", "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Ruim", "Boa", 
      "Ruim", "Ruim", "Boa", "Ruim", "Boa", "Boa", "Boa", "Ruim"]})

      
                                    
#Converte a classe numa binária, para poder fazer a regressão logística
dados['Classe_Cod'] = pd.Categorical(dados['Classe'],categories = ["Ruim","Boa"], ordered = True).codes


# Perform small exploratory analysis
# Group data by 'Classe' and calculate means for all variables
means_by_classe = dados.groupby('Classe').mean()

print(means_by_classe)


#Fórmula
X = dados[['Prova_Logica','Redacao','Auto_Avaliacao']].values
y = dados['Classe_Cod']
                                    
# Fit da regressão logística
model = LogisticRegression()
model.fit(X, y)

#Coeficientes e intercepto
intercepto = model.intercept_
coeficientes = model.coef_

print("Intercepto: ", intercepto)
print("Coeficientes: ", coeficientes)

# Aplica exponenciação nos coeficientes para interpretar
print('Exponencial dos coeficientes:',  np.exp(coeficientes))

print('Exponencial do intercepto:',  np.exp(intercepto))

# Obtem a predicao/probabilidade para cada observacao
Proba = model.predict_proba(X)

Probabilidade =Proba[:,1] #Para pegar apenas a segunda coluna

import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(dados['Classe_Cod'], Probabilidade)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


# Se a probabilidade for maior que 50% classifica como 'Boa'
Classe_Predita = np.where(Probabilidade > 0.5, "Boa", "Ruim")

# Adicionar a classificação predita ao DataFrame
dados['Probabilidade'] = Probabilidade
dados['Classe_Predita'] = Classe_Predita
print(dados[['Probabilidade', 'Classe_Predita']])


# Gera a matriz de confusão
labels =['Boa', 'Ruim']
confusao = confusion_matrix(dados['Classe'], dados['Classe_Predita'], labels=labels)
print("Matriz de Confusão:\n", confusao)

# Converte a matriz de confusão em um DataFrame do pandas para impressão com rótulos
confusao_df = pd.DataFrame(confusao, index=labels, columns=[f'Predito_{label}' for label in labels])
print("Matriz de Confusão:")
print(confusao_df)

# Armazena valores da matriz de confusão
vp = confusao[0, 0]  # Verdadeiros positivos
fn = confusao[0, 1]  # Falsos negativos
vn = confusao[1, 1]  # Verdadeiros negativos
fp = confusao[1, 0]  # Falsos positivos

# Calcula a acurácia
acuracia = (vp + vn) / confusao.sum()
print(f"Acurácia: {acuracia:.3f}")

# Calcula a Sensitividade
sensitividade = vp / (vp + fn)
print(f"Sensitividade: {sensitividade:.3f}")

# Calcula a Especificidade
especificidade = vn / (vn + fp)
print(f"Especificidade: {especificidade:.3f}")




