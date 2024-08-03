#Copie este código, cole na sua IDE e execute para ver os resultados
########## Regressão Linear #############################

import pandas as pd

#Dados a serem trabalhados
dados =pd.DataFrame({"Vendas_Cimento": [2939,2837,2788,3349,2609,2820,3444,2993,3225,2593,2479,2641,3093,2648,2905,3241,2972,3268,2544,2751,2850,3371,2798,2367,3037,2605,3036,3083,3059,3052],
"Preco_Cimento": [94,90,108,81,121,120,80,86,96,114,122,123,92,108,92,90,105,85,127,99,91,91,99,125,96,127,87,82,101,83],
"Propaganda": ['N','N','N','S','N','S','S','N','S','N','N','S','S','N','N','S','S','S','S','N','N','S','N','N','S','S','N','N','S','N'],
"PIB": [1.2,1.4,1.6,1.4,1.9,2.0,1.3,1.5,1.7,1.2,1.2,2.0,1.0,1.2,1.3,2.0,2.0,1.5,1.8,1.6,1.6,1.9,1.4,1.0,1.1,1.6,1.4,1.3,1.0,1.6]
})

print(dados)

#Plotagem para visualizar vendas  x preço 
import matplotlib.pyplot as plt

plt.scatter(x=dados['Preco_Cimento'], y=dados['Vendas_Cimento'], marker='o')
plt.title('Relação entre Vendas do Cimento VS Preço')
plt.xlabel('Preço do Cimento')
plt.ylabel('Qtde Vendida do Cimento')
plt.grid(True)
plt.show()



#Correlacao
correlacao = dados['Vendas_Cimento'].corr(dados['Preco_Cimento'])
print("Correlação entre Vendas_Cimento e Preco_Cimento:", correlacao)

#==========================
#Regressão com o sklearn.linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Criar e treinar o modelo de regressão linear
model = LinearRegression()

X =dados[['Preco_Cimento']].values
Y =dados['Vendas_Cimento'].values

model.fit(X, Y)

# Fazer previsões
Y_pred = model.predict(X)

# Avaliar o modelo
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizar os resultados
plt.scatter(X, Y, color='blue', label='Real')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regressão')
plt.title('Real vs Regressão')
plt.xlabel('Preço de Cimento')
plt.ylabel('Vendas de Cimento')
plt.legend()
plt.show()

#===================
#Para saber os coeficientes do modelo, podemos utilizar um comando como o seguinte
coeficiente = model.coef_[0]  # Inclinação (coeficiente da variável independente)
intercepto = model.intercept_  # Intercepto (constante)

print(f'Coeficiente (Inclinação): {coeficiente}')
print(f'Intercepto: {intercepto}')

#==========================


#Fazendo o mesmo procedimento, para PIB X Vendas Cimento
import matplotlib.pyplot as plt

plt.scatter(x=dados['PIB'], y=dados['Vendas_Cimento'], marker='o')
plt.title('Relação entre Vendas do Cimento VS PIB')
plt.xlabel('PIB')
plt.ylabel('Qtde Vendida de Cimento')
plt.grid(True)
plt.show()


#Coeficiente de correlacao PIB e as vendas do Cimento
correlacao = dados['Vendas_Cimento'].corr(dados['PIB'])
print("Correlação entre Vendas_Cimento e PIB:", correlacao)

#Note que a correlação é baixíssima, o que dá para notar pelo gráfico também

# Criar e treinar o modelo de regressão linear
modelPibCimento = LinearRegression()

XPib =dados[['PIB']].values
YCim =dados['Vendas_Cimento'].values

modelPibCimento.fit(XPib, YCim)

# Fazer previsões
Y_pred_cim = modelPibCimento.predict(XPib)

# Avaliar o modelo
mse = mean_squared_error(YCim, Y_pred_cim)
r2 = r2_score(YCim, Y_pred_cim)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Note que o R2 também é muito baixo, mostrando como o PIB não explica o Volume de vendas.

# Visualizar os resultados
plt.scatter(XPib, YCim, color='blue', label='Real')
plt.plot(XPib, Y_pred_cim, color='red', linewidth=2, label='Regressão')
plt.title('Real vs Regressão')
plt.xlabel('PIB')
plt.ylabel('Vendas de Cimento')
plt.legend()
plt.show()

#Embora seja possível criar um modelo, ele não vai ter poder explicativo algum. Uma forma de verificar isso é pelo R^2 é baixo

#===================
#Separacao em base de treino e de teste
#É uma boa prática separar as bases em treino (para treinar o modelo) e teste (para avaliar como ele funciona numa base que nunca vira antes).
#Isso tem como objetivo evitar o overfitting.
#No sklearn, existe uma função que ajuda a dividir randomicamente a base. É o train_test_split

from sklearn.model_selection import train_test_split

#Vamos partir do modelo de Preco x Vendas
model3 = LinearRegression()

X =dados[['Preco_Cimento']].values
Y =dados['Vendas_Cimento'].values

#Dividindo X e Y em bases de treino e teste. Teste terá 10% dos dados
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

#Treinamento com base de treino
model3.fit(X_train, Y_train)

# Fazer previsões na base de teste
Y_pred = model3.predict(X_test)

# Avaliar o modelo utilizando a base de testes
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizar os resultados
plt.scatter(X, Y, color='blue', label='Real')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regressão')
plt.title('Real vs Regressão')
plt.xlabel('Preço de Cimento')
plt.ylabel('Vendas de Cimento')
plt.legend()
plt.show()

#Como a base de testes tem apenas três números, o plot mostra uma pequena linha vermelha apenas



#===================
#Vamos plotar um Boxplot de vendas do Cimento por Propaganda
#para visualmente verificar se há indícios de que propaganda tem efeito

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.boxplot(x=dados['Propaganda'], y=dados['Vendas_Cimento'])
plt.title('Boxplot das Vendas do Cimento por Propaganda')
plt.xlabel('Propaganda')
plt.ylabel('Vendas do Cimento')
plt.show()


#=======================
#Podemos fazer, a título de exercício, regressão com duas variáveis (Preço e PIB).

# Criar e treinar o modelo de regressão linear
model4 = LinearRegression()

X =dados[['Preco_Cimento','PIB']].values
Y =dados['Vendas_Cimento'].values

model4.fit(X, Y)

# Fazer previsões
Y_pred = model4.predict(X)

# Avaliar o modelo
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



#Gráfico simples comparando resultados
plt.scatter(range(len(Y)), Y, color='blue', label='Real')
plt.scatter(range(len(Y_pred)), Y_pred, color='red', marker='x', label='Regressão')
plt.title('Real vs Regressão')
plt.xlabel('Índice da Venda')
plt.ylabel('Vendas')
plt.legend()
plt.show()


#O pacote scikit learn não prove diretamente o r2 ajustado, mas é possível utilizar uma funcao como a seguinte
def adjusted_r2_score(y_true, y_pred, num_predictors):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)

mse_ajustado = adjusted_r2_score(Y, Y_pred,2) #2 variaveis de predicao
print(f'R2 ajustado: {mse_ajustado}')

#===============================
#Bônus. Há diversos outros pacotes possíveis, como o statsmodels

import statsmodels.api as sm
# Adicionando uma coluna numérica para representar 'Propaganda'
dados['Propaganda_Num'] = dados['Propaganda'].apply(lambda x: 1 if x == 'Sim' else 0)

# Definindo as variáveis independentes e dependentes
X = dados[['Preco_Cimento', 'PIB', 'Propaganda_Num']]
y = dados['Vendas_Cimento']

# Adicionando uma constante para o termo independente
X = sm.add_constant(X)

# Criando e ajustando o modelo de regressão linear
modelo = sm.OLS(y, X).fit()

# Imprimindo o resumo do modelo
print(modelo.summary())


