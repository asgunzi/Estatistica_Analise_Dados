#Qual o número de possibilidades de resultados ao jogar 6 números na mega sena?

#Código de combinações
import math


# Número total de elementos (n)
n = 3
# Número de elementos a serem escolhidos (k)
k = 2

# Calcula o número de combinações
combinacoes_possiveis = math.comb(n, k)

print(f"O número de combinações é: {combinacoes_possiveis}")


#==============

# Número total de elementos (n)
n = 60
# Número de elementos a serem escolhidos (k)
k = 6

# Calcula o número de combinações
combinacoes_possiveis = math.comb(n, k)

print(f"O número de possibilidades ao escolher 6 números na Mega-Sena é: {combinacoes_possiveis:,}")

#====================
#E se eu jogar 10 números?

import math

# Número total de elementos (n)
n = 60

# Número de elementos a serem escolhidos (k)
k = 6

#Números jogados
njogados = 10

# Calcula o número de combinações das jogadas 
combinacoes_jogadas = math.comb(njogados, k)

print(f"O número de combinar {njogados} números em grupos de 6 é: {combinacoes_jogadas}")


# Calcula o número de combinações totais possíveis
combinacoes_possiveis = math.comb(n, k)

print(f"A probabilidade de vitória com {njogados} números é de {combinacoes_jogadas} em : {combinacoes_possiveis}")


#====================

# Ao passar 10 clientes em nossa loja, qual a probabilidade de realizarmos 2 vendas?
#Ou seja, queremos encontrar a probabilidade de dois sucessos, em dez tentativas. Cuja probabilidade de sucesso
# em cada tentativa é 50%

from scipy.stats import binom

# Definindo os parâmetros
x = 2 #sucessos 
size = 10 #Tentativas
prob = 0.5 #Prob sucesso

# Calculando a probabilidade
probability = binom.pmf(x, size, prob)

print(probability)

#Onde:
# x é o número de sucessos,
# size é o número de tentativas,
# prob é a probabilidade de sucesso em cada tentativa


#======================

#Probabilidade, Probabilidade acumulada e amostragem
#A biblioteca stats utiliza pmf para probabilidade, cdf para probabilidade acumulada e rvs para uma amostra segundo a variável aleatória

#binom.pmf(k, n, p) calcula a probabilidade de obter mais de k sucessos em n tentativas com a probabilidade p de sucesso em cada tentativa.

#binom.cdf(q, n, p) calcula o número de sucessos correspondentes à probabilidade cumulativa q em n tentativas com a probabilidade p de sucesso em cada tentativa.

#binom.rvs(n, p, size) gera amostras aleatórias seguindo uma distribuição binomial com n tentativas e probabilidade p de sucesso. Size refere-se ao número de amostras desejado.

from scipy.stats import binom

# Definindo os parâmetros
k = 2
n = 10
prob = 0.5

# Calcula a probabilidade de 2 sucessos em 10 jogadas, com probabilidade 0,5 de sucesso
probability = binom.pmf(k, n, prob)
print("Probabilidade de k sucessos em 10 lançamentos: ", probability)


# Calculando a probabilidade acumulada de 2 sucessos em 10 jogadas (ou seja, de 0, 1 e 2 sucessos), com probabilidade 0,5 de sucesso
print("Probabilidade acumulada: ",binom.cdf(k, n, prob))

#É a mesma coisa que somar as chances de 0, 1 e 2 sucessos
print("Somando chances de 0, 1 e 2 sucessos: ", binom.pmf(0, n, prob) + binom.pmf(1, n, prob) + binom.pmf(2, n, prob))

#======================
#Exemplo de geração de números aleatórios a partir de uma variável aleatória

#Vamos jogar uma moeda e verificar quantas caras ocorreram
from scipy.stats import bernoulli

# Definindo a probabilidade de sucesso (p)
p = 0.5 # Por exemplo, p = 0.5 significa 50% de chance de obter 1

# Gerando 15 números a partir da distribuição de Bernoulli
n = 15

sorteios = bernoulli.rvs(p, size=n)

# Exibindo os resultados
print("Números sorteados:", sorteios)


#======================

#A fim de mostrar que há outras bibliotecas estatísticas, vamos utilizar a biblioteca numpy para gerar os números aleatórios segundo uma binomial e o matplotlib para plotar

import numpy as np
import matplotlib.pyplot as plt

# Gerando os números binomiais
va_binomial = np.random.binomial(n=10, p=0.5, size=30)

# Criando o histograma
plt.hist(va_binomial, bins=range(min(va_binomial), max(va_binomial) + 2), edgecolor='black')
plt.xlabel('Número de Sucessos')
plt.ylabel('Frequência')
plt.title('Histograma de Variáveis Aleatórias Binomiais')
plt.show()


##################################
#### DISTRIBUIÇÃO GEOMÉTRICA ####
##################################
#Exemplo: Definindo como sucesso o cliente comprar, e supondo que a probabilidade de sucesso é 50%.
#Qual a probabilidade da primeira venda ocorrer quando o quinto cliente entrar na loja?
from scipy.stats import geom

# Definindo os parâmetros
x = 5
prob = 0.5

# Calculando a probabilidade
probability = geom.pmf(x, prob)

print(probability)

#Onde:
# x é o número de tentativas até o primeiro sucesso (incluindo o sucesso)
# prob é a probabilidade de sucessos


# Podemos utilizar a mesma função para nos dar a probabilidade do sucesso ocorrer na primeira tentativa,
#Segunda tentativa, terceira tentativa ... até a décima tentativa.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# Definindo os parâmetros
x = np.arange(1, 11)
prob = 0.5

# Calculando as probabilidades
va_geometrica = geom.pmf(x, prob)

# Exibindo as probabilidades
print(va_geometrica)

# Criando o gráfico
plt.plot(x, va_geometrica, marker='o')
plt.xlabel('Número de Tentativas')
plt.ylabel('Probabilidade')
plt.title('Distribuição Geométrica (p = 0.5)')
plt.grid(True)
plt.show()

#Veja como as probabilidades vão diminuindo.
#então é muito provavel que o sucesso ocorra logo nas primeiras tentativas

#======================
#======================
# Podemos utilizar a distribuição geométrica acumulada para saber qual a probabilidade do  primeiro sucesso
#ocorrer na primeira tentativa OU na segunda tentativa OU na terceira tentativa

#Ou seja queremos: P(X<=3)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# Definindo os parâmetros
x = np.arange(0, 4)
prob = 0.5

# Calculando as probabilidades acumuladas
va_geometrica_acumulada = geom.cdf(x, prob)

# Exibindo as probabilidades acumuladas
print(va_geometrica_acumulada)

# Criando o gráfico
plt.plot(x, va_geometrica_acumulada, marker='o')
plt.xlabel('Número de Falhas antes do Primeiro Sucesso')
plt.ylabel('Probabilidade Acumulada')
plt.title('Distribuição Geométrica Acumulada (p = 0.5)')
plt.grid(True)
plt.show()


#########################################
#### DISTRIBUIÇÃO BINOMIAL NEGATIVA ####
#########################################
# Exemplo: Definindo como sucesso o cliente comprar, e supondo que a probabilidade de sucesso é 50%. 
#Qual a probabilidade de ter que entrar 8 clientes até que a segunda venda ocorra?
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# Definindo os parâmetros
x = np.arange(0, 4)
prob = 0.5

# Calculando as probabilidades acumuladas
va_geometrica_acumulada = geom.cdf(x, prob)

# Exibindo as probabilidades acumuladas
print(va_geometrica_acumulada)

# Criando o gráfico
plt.plot(x, va_geometrica_acumulada, marker='o')
plt.xlabel('Número de Falhas antes do Primeiro Sucesso')
plt.ylabel('Probabilidade Acumulada')
plt.title('Distribuição Geométrica Acumulada (p = 0.5)')
plt.grid(True)
plt.show()

#Onde:
# x é o número de sucessos
# size é a quantidade de tentativos
# prob é a probabilidade de sucesso

#########################################
#### DISTRIBUIÇÃO POISSON ####
#########################################
# Exemplo: Uma loja recebe em média, 6 (𝝺) clientes por minuto. Qual a probabilidade de que 5(x) clientes
#entrem em um minuto? 

from scipy.stats import poisson

# Definindo os parâmetros
x = 5
lam = 6

# Calculando a probabilidade
probability = poisson.pmf(x, lam)

print(probability)

#Onde:
# x é a quantidade a ser testada


# lambda é a taxa média de ocorrêcia do evento em um determinado período de intervalo de tempo ou espaço 


# Podemos utilizar a mesma funcao para obter a probabilidade de entrar um cliente, dois clientes... quinze clientes

# Observe que os valores se distribuem simetricamente en tormo de seis, use acontece porque o parametro
#lambda é a média (e também o desvio padrão) da distribuição de Poisson
# Também podemos obter a probabilidade acumulada de até 5 clientes entrarem na loja em  um minuto
#Formalizando, queremos: P(X<=5)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Definindo os parâmetros
x = np.arange(1, 16)
lam = 6

# Calculando as probabilidades
va_poisson = poisson.cdf(x, lam)

# Exibindo as probabilidades
print(va_poisson)

# Criando o gráfico
plt.plot(x, va_poisson, marker='o')
plt.xlabel('Número de Eventos')
plt.ylabel('Probabilidade Acumulada')
plt.title('Distribuição de Poisson (λ = 6)')
plt.grid(True)
plt.show()



#########################################
#### DISTRIBUIÇÃO NORMAL ####
#########################################
# Exemplo: Suponha que a distribuição dos salários dos funcionários de uma empresa sigam uma distribuição
#normal com média 𝜇=2.500 e desvio padrão σ= 170.
# Ao selecionar aleatoriamente um indivíduo dessa população, qual a probabilidade de ter salário entre 
#2.400 e 2.600 ?
# Precisamos achar a probabilidade do indivíduo ter um salário de até 2.600 e subtrair pela  probabilidade do
#indivíduo ter o salário até 2.400

from scipy.stats import norm

# Definindo os parâmetros
mean = 2500
sd = 170

# Calculando as probabilidades acumuladas
probabilidade_ate_2600 = norm.cdf(2600, mean, sd)
probabilidade_ate_2400 = norm.cdf(2400, mean, sd)

# Calculando a diferença das probabilidades
diferenca_probabilidade = probabilidade_ate_2600 - probabilidade_ate_2400

print(diferenca_probabilidade)


#Podemos gerar 100 números aleatórios para uma distribuição normal com média 2500 e desvio padrão 170

import numpy as np
import matplotlib.pyplot as plt

# Definindo os parâmetros
n = 100
mean = 2500
sd = 170

# Gerando os números da distribuição normal
va_normal = np.random.normal(mean, sd, n)

# Criando o histograma
plt.hist(va_normal, bins=10, edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.title('Histograma de Variáveis Aleatórias Normais')
plt.grid(True)
plt.show()



#########################################
#### DISTRIBUIÇÃO F ####
#########################################
#Gerando uma amostra aleatória de 1000 número seguindo uma distribuição F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# Definindo os parâmetros
n = 1000
df1 = 5
df2 = 33

# Gerando os números da distribuição F
va_f = f.rvs(df1, df2, size=n)

# Criando o histograma
plt.hist(va_f, bins=30, edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.title('Histograma de Variáveis Aleatórias F')
plt.grid(True)
plt.show()

#Vá aumentando os graus de liberdade e observe como a distribuição se aproxima da normal
#Informação Extra: Uma distribuição F é a razão entre duas chi-quadrado


#########################################
#### DISTRIBUIÇÃO T ####
#########################################
#Gera uma amostra aleatória de 1000 números seguindo uma distribuição T
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Definindo os parâmetros
n = 1000
df = 2

# Gerando os números da distribuição t
va_t = t.rvs(df, size=n)

# Criando o histograma
plt.hist(va_t, bins=30, edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.title('Histograma de Variáveis Aleatórias t (df = 2)')
plt.grid(True)
plt.show()

#Observe que a distribuição t, assim como a normal padrão, é centrada no zero
#Vá aumentando o grau de liberdade e observando o comportamento do histograma

#########################################
#### DISTRIBUIÇÃO QUI-QUADRADO ####
#########################################
#Gera uma amostra aleatória de 1000 números seguindo uma distribuição qui-quadrado
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Definindo os parâmetros
n = 1000
df = 3

# Gerando os números da distribuição qui-quadrado
va_QuiQuadrado = chi2.rvs(df, size=n)

# Criando o histograma
plt.hist(va_QuiQuadrado, bins=30, edgecolor='black')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.title('Histograma de Variáveis Aleatórias Qui-Quadrado (df = 3)')
plt.grid(True)
plt.show()



