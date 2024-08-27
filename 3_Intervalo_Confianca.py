#Copie este código, cole na sua IDE e execute para ver os resultados
########## Intervalo de confiança para média amostral pela distribuição Normal Padrão #############################
# Obter o intervalo de confiança para uma variável cuja média = 30, desvio padrão = 7,31 e n = 30
#Temos que definir o nível de confiança do nosso intervalo.
#Podemos obter o valor do quantil para o nível de confiança desejado com a função qnorm()
#O quantil na distribuição normal padrão para 95% de confiança

import scipy.stats as stats
import math

# Definindo o nível de confiança
ic = 0.95
alfa = 1 - ic
z = stats.norm.ppf(1 - alfa / 2)
#O "ppf" é de "percentil". O código está calculando o percentil 97,5% (a fim de ter 2,5% para cada lado)

# Definindo os parâmetros
media = 30
desvio_padrao_populacional = 7.31
n = 30

# Calculando os limites do intervalo de confiança
limite_superior = media + z * (desvio_padrao_populacional / math.sqrt(n))
limite_inferior = media - z * (desvio_padrao_populacional / math.sqrt(n))

# Exibindo o resultado
print(f"Com 95% de confiança, a média varia entre {limite_inferior:.2f} e {limite_superior:.2f}")


####################### Intervalo de confiança para a média amostral pela  distribuição t de Student ############
#A teoria nos diz para utilizar a distribuição t de Student quando não soubermos o desvio padrão populacional.
#Vamos assumir que o desvio padrão que temos é obtido a partir da amostra
#Vamos armazenar os valores em objetos

import scipy.stats as stats
import math

# Definindo os parâmetros
media = 30
desvio_padrao_amostral = 7.31
n = 30

# Calculando o quantil t para 95% de confiança
quantil_95_t = stats.t.ppf(0.975, df=n-1)

# Calculando os limites do intervalo de confiança
limite_superior_t = media + quantil_95_t * (desvio_padrao_amostral / math.sqrt(n))
limite_inferior_t = media - quantil_95_t * (desvio_padrao_amostral / math.sqrt(n))

# Exibindo o resultado
print(f"Com 95% de confiança, a média varia entre {limite_inferior_t:.2f} e {limite_superior_t:.2f}")



#=======================
#Vamos gerar uma variável aleatoria com média 30, desvio padrão 7,31 e n = 30


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Gerando dados aleatórios
np.random.seed(0)  # Define a semente para reprodutibilidade
n = 30
media = 30
desvio_padrao = 7.31
va = np.random.normal(loc=media, scale=desvio_padrao, size=n)

# Visualizando o histograma
plt.hist(va, bins=10, edgecolor='black')
plt.title('Histograma de va')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

# Calculando o intervalo de confiança de 95% usando a distribuição t de Student
IC = stats.t.interval(0.95, df=n-1, loc=np.mean(va), scale=stats.sem(va))

print(f"Intervalo de confiança de 95%: {IC}")


####################### Intervalo de confiança para a proporção 
############
#Utilizando o exemplo da apostila, onde calculamos o intervalo para proporção onde
# 138 de n = 500 clientes realizaram a devolução do produto
#Vamos armazenar os valores em objetos

import scipy.stats as stats
import math

# Definindo os parâmetros
devolucoes = 138
n = 500
proporcao_devolucoes = devolucoes / n

# Calculando o quantil z para 95% de confiança
quantil_95 = stats.norm.ppf(0.975)

# Calculando os limites do intervalo de confiança
erro_padrao = math.sqrt(proporcao_devolucoes * (1 - proporcao_devolucoes) / n)
limite_superior_prop = proporcao_devolucoes + quantil_95 * erro_padrao
limite_inferior_prop = proporcao_devolucoes - quantil_95 * erro_padrao

# Exibindo o resultado
print(f"Com 95% de confiança, podemos afirmar que a proporção varia entre {limite_inferior_prop:.4f} e {limite_superior_prop:.4f}")




############### Intervalo de confiança para média via Bootstrap 
############
#Vamos gerar uma va seguindo uma distribuição qui-quadrado

import numpy as np
import matplotlib.pyplot as plt

# Gerando a amostra original com distribuição qui-quadrado
np.random.seed(0)  # Define a semente para reprodutibilidade
n = 60
df = 3
va = np.random.chisquare(df, n)

# Visualizando a distribuição da amostra original
plt.hist(va, bins=20, edgecolor='black')
plt.title('Distribuição da amostra original (va)')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

# Inicializando variáveis
R = 1000  # Número de subamostras para gerar a distribuição amostral das médias
medias = []

# Bootstrap
for _ in range(R):
    reamostra = np.random.choice(va, size=50, replace=True)
    medias.append(np.mean(reamostra))

# Visualizando a distribuição das médias das subamostras
plt.hist(medias, bins=20, edgecolor='black')
plt.title('Distribuição das médias das subamostras (bootstrap)')
plt.xlabel('Média')
plt.ylabel('Frequência')
plt.show()


#==============================
#Vamos realizar mais um experimento

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Gerando a amostra original
np.random.seed(0)  # Define a semente para reprodutibilidade
n = 30
media = 30
desvio_padrao = 7.31
va = np.random.normal(loc=media, scale=desvio_padrao, size=n)

# Inicializando variáveis
R = 10000  # Número de subamostras para gerar a distribuição amostral das médias
medias = []

# Bootstrap
for _ in range(R):
    reamostra = np.random.choice(va, size=20, replace=True)
    medias.append(np.mean(reamostra))

# Visualizando a distribuição das médias das subamostras
plt.hist(medias, bins=30, edgecolor='black')
plt.title('Distribuição das médias das subamostras (bootstrap)')
plt.xlabel('Média')
plt.ylabel('Frequência')
plt.show()

# Calculando o intervalo de confiança usando a distribuição t de Student
media_bootstrap = np.mean(medias)
desvio_padrao_bootstrap = np.std(medias, ddof=1)
intervalo_conf = stats.t.interval(0.95, df=R-1, loc=media_bootstrap, scale=desvio_padrao_bootstrap/np.sqrt(R))

# Exibindo o intervalo de confiança
print(f"Intervalo de confiança de 95% usando a distribuição t de Student: {intervalo_conf}")



#=========================
#Experimento para ilustrar o Teorema do Limite Central
import random
import matplotlib.pyplot as plt

#Imagine que eu jogo uma moeda para cima, e veja o resultado.

#Só que é uma moeda viciada, e Probabilidade de dar 0 é dada abaixo
p = 0.9 

#Se eu rodar o código a seguir, simulo o resultado
sorteio = random.random()
if sorteio < p:
    resultado =0
else:
    resultado =1

print(resultado)



#Agora, imagine que eu sorteie a mesma moeda viciada s_sorteios vezes, e some o número de 1's que ocorrerem 

s_sorteios = 50 #Sorteios a somar por experimento

n_experimentos = 1000

p = 0.9 #Probabilidade de dar 0

experimentos = []

for n in range(n_experimentos):
    soma =0    
    for s in range(s_sorteios):
        sorteio = random.random()
        if sorteio < p:
            resultado =0
        else:
            resultado =1
        soma += resultado    
    experimentos.append(soma)

plt.hist(experimentos ,bins = 10)
plt.title("Histograma de Resultados")
plt.xlabel("Valor")
plt.ylabel("Frequência")

#-----------------------------
#Pergunta: A pontuação dos alunos em um teste segue uma distribuição normal com média 75 pontos e desvio padrão de 10 pontos. Qual a probabilidade de um aluno escolhido ao acaso ter uma pontuação entre 70 e 80 pontos?


import scipy.stats as stats

# Dados do problema
mu = 75  # média
sigma = 10  # desvio padrão

# Pontuações para as quais queremos calcular a probabilidade
x1 = 70
x2 = 80

# Calculando a probabilidade usando a função de distribuição acumulada (CDF)
p1 = stats.norm.cdf(x1, mu, sigma)
p2 = stats.norm.cdf(x2, mu, sigma)

# Probabilidade de um aluno ter pontuação entre 70 e 80 pontos
probabilidade = p2 - p1
print(probabilidade)



