####### Análise exploratória de dados #######

import pandas as pd

#Copie este código, cole na sua IDE favorita e execute para ver os resultados

#Cria o data frame contendo o historico de vendas do cafe
dados = pd.DataFrame({
    'Vendas_Cafe': [18, 20, 23, 23, 23, 23, 24, 25, 26, 26, 26, 26, 27, 28, 28,
                    29, 29, 30, 30, 31, 31, 33, 34, 35, 38, 39, 41, 44, 44, 46],
    'Preco_Cafe': [4.77, 4.67, 4.75, 4.74, 4.63, 4.56, 4.59, 4.75, 4.75, 4.49,
                   4.41, 4.32, 4.68, 4.66, 4.42, 4.71, 4.66, 4.46, 4.36, 4.47, 4.43,
                   4.4, 4.61, 4.09, 3.73, 3.89, 4.35, 3.84, 3.81, 3.79],
    'Promocao': ["Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Sim",
                 "Nao", "Sim", "Nao", "Nao", "Sim", "Sim", "Nao", "Sim", "Sim",
                 "Sim", "Nao", "Nao", "Sim", "Sim", "Sim", "Nao", "Sim", "Sim",
                 "Sim", "Sim", "Sim"],
    'Preco_Leite': [4.74, 4.81, 4.36, 4.29, 4.17, 4.66, 4.73, 4.11, 4.21, 4.25,
                    4.62, 4.53, 4.44, 4.19, 4.37, 4.29, 4.57, 4.21, 4.77, 4, 4.31,
                    4.34, 4.05, 4.73, 4.07, 4.75, 4, 4.15, 4.34, 4.15]
})

#Visualização dos dados
print(dados)

#Para visualizar somente uma coluna de dados, utilize comando similar a:
print(dados['Preco_Leite'])

#--------------
#Estatística descritiva utilizando pandas

#Há diversas formas de fazer o cálculo. No Pandas, vamos utilizar a funçao .mean()
# Calcular a média da coluna 'Vendas_Cafe'
media_preco_cafe = dados['Preco_Cafe'].mean()

print("Média do preço de Café:", media_preco_cafe)

# Calcular o desvio padrão da coluna 'Vendas_Cafe'
desvpad_preco_cafe = dados['Preco_Cafe'].std()
print("Desvio Padrão do preço de Café:", desvpad_preco_cafe)

# O mesmo para outra coluna, Preco_Leite
desvpad_preco_leite = dados['Preco_Leite'].std()
print("Desvio Padrão do preço do Leite:", desvpad_preco_leite)


# Utilize o comando describe() do pandas para um sumário de estatísticas
#Informações descritas: contagem, média, desvio padrão, mínimo, percentil 25%, percentil 50% (mediana), percentil 75%, máximo
print(dados['Preco_Cafe'].describe())


#----------------------------
#O pandas não é o único pacote disponível, na verdade, há vários outros possíveis. Vamos nos ater aos mais simples.
#Utilizando o statistics

import statistics


# Calcular a média
media = statistics.mean(dados['Preco_Cafe'])
print("Média:", media)

# Calcular a mediana
mediana = statistics.median(dados['Preco_Cafe'])
print("Mediana:", mediana)

# Calcular a variância
variancia = statistics.variance(dados['Preco_Cafe'])
print("Variância:", variancia)


# Calcular o desvio padrão
desvio_padrao = statistics.stdev(dados['Preco_Cafe'])
print("Desvio Padrão:", desvio_padrao)

# Calcular a amplitude
amplitude = max(dados['Preco_Cafe']) - min(dados['Preco_Cafe'])
print("Amplitude:", amplitude)

# Calcular a moda
print("A moda (valor mais frequente) é:",statistics.mode(dados['Preco_Cafe']))


#--------------
#--------------

#Gráficos utilizados
#Vamos utilizar a biblioteca Matplotlib
import matplotlib.pyplot as plt

# Plotar o histograma da coluna 'Vendas_Cafe'
plt.hist(dados['Preco_Cafe'], bins=6, edgecolor='black')

# Adicionando títulos e rótulos
plt.title('Distribuição do preço de Café')
plt.xlabel('Preço do Café')
plt.ylabel('Frequência')


# Mostrar o gráfico
plt.show()

#--------------
#Plotando diversos subgráficos com cada histograma em um subgráfico

# Criar subplots
fig, axs = plt.subplots(3, figsize = (4,5))  # 3 linhas, 1 coluna

# Primeiro subplot
axs[0].hist(dados['Vendas_Cafe'], bins = 6, edgecolor = 'black')
axs[0].set_title('Distribuição de Vendas de Café')
axs[0].set_xlabel('Vendas de Café')
axs[0].set_ylabel('Frequência')


# Segundo subplot
axs[1].hist(dados['Preco_Cafe'], bins = 6, color = 'darkgreen', edgecolor = 'black')
axs[1].set_title('Distribuição do Preço de Café')
axs[1].set_xlabel('Preço do Café')
axs[1].set_ylabel('Frequência')

# Terceiro subplot
axs[2].hist(dados['Preco_Leite'], bins = 6, color = 'purple', edgecolor = 'white')
axs[2].set_title('Distribuição do Preço do Leite')
axs[2].set_xlabel('Preço do Leite')
axs[2].set_ylabel('Frequência')

plt.tight_layout()

# Mostrar o plot
plt.show()


#--------------
#Scatter plot
# Adicionando títulos e rótulos
plt.title('Relação entre Preço do Café e Vendas de Café')
plt.xlabel('Preço do Café')
plt.ylabel('Vendas de Café')

plt.scatter(dados['Preco_Cafe'], dados['Vendas_Cafe'])

# Mostrar o gráfico
plt.show()


#--------------
#Separando dataframe com e sem promocao
dadosNao = dados [dados['Promocao']=="Nao"]

dadosSim = dados [dados['Promocao'] =="Sim"]

print(dadosNao)

print(dadosSim)


plt.title('Relação entre Preço do Café e Vendas de Café')
plt.xlabel('Preço do Café')
plt.ylabel('Vendas de Café')

plt.scatter(dadosSim['Preco_Cafe'], dadosSim['Vendas_Cafe'], color = 'red', label = 'Com promoção')
plt.scatter(dadosNao['Preco_Cafe'], dadosNao['Vendas_Cafe'], color = 'blue', label='Sem promoção')

plt.legend()


# Mostrar o gráfico
plt.show()

#--------------
# Gerar boxplot das vendas de café
plt.boxplot(dados['Vendas_Cafe'])

# Adicionar títulos e rótulos
plt.title('Boxplot das Vendas de Café')
plt.ylabel('Vendas de Café')

# Configurar os rótulos do eixo x para mostrar que representam as Vendas de Café
plt.xticks([1], ['Vendas de Café'])
plt.show()


#Mesma coisa para Preço do Café
plt.boxplot(dados['Preco_Cafe'])

# Adicionar títulos e rótulos
plt.title('Boxplot dos Preços de Café')
plt.ylabel('Preços de Café')

# Configurar os rótulos do eixo x para mostrar que representam as Vendas de Café
plt.xticks([1], ['Preços de Café'])
plt.show()


#--------------

#Plota boxplot com dados de promocao Não e sim
plt.boxplot([dadosNao['Vendas_Cafe'], dadosSim['Vendas_Cafe']])
plt.title('Boxplot das Vendas de Café por Promoção')
plt.xticks([1,2], ['Não','Sim'])
plt.ylabel('Vendas de Café')
plt.show()


