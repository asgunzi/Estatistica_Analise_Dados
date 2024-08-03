#Copie este código, cole na sua IDE e execute para ver os resultados
########## Teste de hipóteses #############################

#Vamos visualizar o QQ plot de uma v.a. normal e uma não normal
#e fazer um teste de hipóteses de normalidade shapiro-wilk
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configura a semente para reprodutibilidade
np.random.seed(11)

# Gera v.a. que segue distribuição normal com n = 70, média = 25 e desvio padrão = 8
va_normal = np.random.normal(loc=25, scale=8, size=70)

# Gera v.a. que segue uma distribuição F (não normal) com n = 15, 2 graus de liberdade no numerador e 10 graus de liberdade no denominador
va_nao_normal = np.random.f(dfnum=2, dfden=10, size=15)

#Outro teste, envolvendo variável uniforme
#va_nao_normal = np.random.uniform(size=15)*25

# Visualize o histograma das variáveis geradas
# Observe como os dados se distribuem em torno do valor médio na va normal
plt.hist(va_normal, bins=10, edgecolor='black')
plt.title('Histograma de va_normal')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

# Observe como os dados não se distribuem em torno de um valor médio exibindo padrão assimétrico
plt.hist(va_nao_normal, bins=10, edgecolor='black')
plt.title('Histograma de va_nao_normal')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

# Visualize o QQ-Plot
# Observe como os pontos de dados da va_normal seguem a linha reta
stats.probplot(va_normal, dist="norm", plot=plt)
plt.title('QQ-Plot de va_normal')
plt.show()



# Observe como os pontos de dados não seguem a linha reta na va não normal
stats.probplot(va_nao_normal, dist="norm", plot=plt)
plt.title('QQ-Plot de va_nao_normal')
plt.show()

#===================================
# Vamos aplicar o teste de hipóteses Shapiro-Wilk
# H0: A variável segue uma distribuição normal
# H1: A variável não segue uma distribuição normal
shapiro_normal = stats.shapiro(va_normal)
shapiro_nao_normal = stats.shapiro(va_nao_normal)

print("Shapiro-Wilk Teste para va_normal:")
print(f"Estatística: {shapiro_normal[0]}, p-valor: {shapiro_normal[1]}")

print("Shapiro-Wilk Teste para va_nao_normal:")
print(f"Estatística: {shapiro_nao_normal[0]}, p-valor: {shapiro_nao_normal[1]}")

# Fixe um nível de significância alfa e analise o p valor (p-value) do Shapiro Wilk
#Se o p-value for menor que alfa a hipótese nula deve ser rejeitada


#====================================
################ Teste t para diferença de médias (duas amostras independentes) 
################
#Iremos testar se:
# H0: As vendas na posição A são iguais as Vendas na Posição B
# H1: As vendas na posição A são diferentes das vendas na posição B

# Define as variáveis
mu1 = 150.1 # Média de vendas na posição A
mu2 = 182.1 # Média de vendas na posição B
s1 = 17.0 # Desvio padrão das vendas na posição A
s2 = 19.2 # Desvio padrão das vendas na posição B
n1 = 25 # Quantidade de observações registradas para vendas na posição A
n2 = 30 # Quantidade de observações registradas para vendas na posição B


from scipy import stats

# Gerando dados simulados
np.random.seed(10)
vendas_A = np.random.normal(loc=mu1, scale=s1, size=n1)
vendas_B = np.random.normal(loc=mu2, scale=s2, size=n2)

#Para visualizar a diferença entre os histogramas das distribuições
import matplotlib.pyplot as plt

plt.hist(vendas_A , bins=10, edgecolor='black')
plt.title('Vendas A')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.hist(vendas_B, bins=10, color='green', edgecolor='black')
plt.title('Vendas B')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# Realiza o teste t de duas amostras independentes
t_test_result = stats.ttest_ind(vendas_A, vendas_B, equal_var=False)
print(f"Estatística t: {t_test_result.statistic}, Valor p: {t_test_result.pvalue}")



################ Teste t para diferença de médias (duas amostras dependentes) 
################
# Amostras dependentes ocorrem quando as observações em uma amostra estão relacionadas às observações em outra amostra. Em outras palavras, as amostras são emparelhadas ou vinculadas de alguma forma: antes e depois de uma dieta, por exemplo
# H0: O peso médio após a dieta é igual ao peso médio antes da dieta
# H1: O peso médio após a dieta é menor do que o peso médio antes da dieta

import numpy as np
from scipy.stats import truncnorm, ttest_rel, shapiro
import matplotlib.pyplot as plt

#Semente para reprodutibilidade
np.random.seed(100)

# Função para trucar normal entre low e upp
def truncated_normal(mean=0, sd=1, low=0, upp=10, size=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size)

# Data before and after diet
antes_da_dieta = truncated_normal(mean=123, sd=18, low=100, upp=140, size=20)
depois_da_dieta = truncated_normal(mean=110, sd=28, low=110, upp=130, size=20)

# Difference
diferenca = depois_da_dieta - antes_da_dieta

# Histograma da diferença
plt.hist(diferenca, bins='auto', edgecolor='black')
plt.title('Distribuição das diferenças de peso')
plt.xlabel('Diferenças de peso')
plt.ylabel('Frequência')
plt.show()

# Shapiro-Wilk test
shapiro_test = shapiro(diferenca)
print("Shapiro-Wilk test for normality:", shapiro_test)

# t-test pareado, utilizando a função ttest_rel do scipy
t_test = ttest_rel(depois_da_dieta, antes_da_dieta, alternative='less')
print("Paired t-test:", t_test)




################ Teste Qui-Quadrado para associação entre variáveis categóricas 
################
# H0: O fato do cliente estar ou não com criança não tem relação com o fato de comprar ou não comprar
# H1: O fato do cliente estar com a criança tem relação com comprar


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2
#Vamos gerar um data frame contendo os dados da pesquisa

# Dados
dados = pd.DataFrame({
    'Cliente': ["Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto", "Adulto", "Adulto", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto", "Adulto", "Adulto", "Adulto",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto", "Adulto_com_Crianca", "Adulto",
                "Adulto", "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto", "Adulto_com_Crianca", "Adulto", "Adulto", "Adulto",
                "Adulto", "Adulto", "Adulto", "Adulto", "Adulto", "Adulto"],
    'Comprou': ["Não_Comprou", "Não_Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Não_Comprou", "Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Não_Comprou", "Comprou", "Não_Comprou", "Comprou",
                "Comprou", "Não_Comprou", "Não_Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Comprou", "Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Não_Comprou", "Não_Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Comprou"]
})

# Visualiza o conjunto de dados
print(dados)

# Gera tabela de contingência
tabela = pd.crosstab(dados['Cliente'], dados['Comprou'])
print(tabela)

# Cria o gráfico de barras
tabela.plot(kind='bar', stacked=True)
plt.xlabel('Cliente')
plt.ylabel('Contagem')
plt.title('Tabela de Contingência')
plt.show()

# Valor crítico para uma distribuição qui-quadrado com 1 grau de liberdade ao nível de confiança de 95%
valor_critico = chi2.ppf(0.95, df=1)
print("Valor crítico:", valor_critico)

# O valor p unilateral
valor_p = 1 - chi2.cdf(10.728, df=1)
print("Valor p:", valor_p)

# Teste qui-quadrado
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2

# Dados
dados = pd.DataFrame({
    'Cliente': ["Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto", "Adulto", "Adulto", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto", "Adulto", "Adulto", "Adulto",
                "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto_com_Crianca", "Adulto", "Adulto_com_Crianca", "Adulto",
                "Adulto", "Adulto_com_Crianca", "Adulto_com_Crianca", "Adulto_com_Crianca",
                "Adulto", "Adulto_com_Crianca", "Adulto", "Adulto", "Adulto",
                "Adulto", "Adulto", "Adulto", "Adulto", "Adulto", "Adulto"],
    'Comprou': ["Não_Comprou", "Não_Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Não_Comprou", "Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Não_Comprou", "Comprou", "Não_Comprou", "Comprou",
                "Comprou", "Não_Comprou", "Não_Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Comprou", "Comprou", "Não_Comprou", "Não_Comprou",
                "Não_Comprou", "Não_Comprou", "Não_Comprou", "Comprou", "Comprou",
                "Comprou", "Comprou", "Comprou", "Comprou"]
})

# Visualiza o conjunto de dados
print(dados)

# Gera tabela de contingência
tabela = pd.crosstab(dados['Cliente'], dados['Comprou'])
print(tabela)

# Cria o gráfico de barras
tabela.plot(kind='bar', stacked=True)
plt.xlabel('Cliente')
plt.ylabel('Contagem')
plt.title('Tabela de Contingência')
plt.show()

# Valor crítico para uma distribuição qui-quadrado com 1 grau de liberdade ao nível de confiança de 95%
valor_critico = chi2.ppf(0.95, df=1)
print("Valor crítico:", valor_critico)

# O valor p unilateral
valor_p = 1 - chi2.cdf(10.728, df=1)
print("Valor p:", valor_p)

# Teste qui-quadrado
chi2_stat, p_val, dof, ex = chi2_contingency(tabela, correction=False)

print(f"Estatística Qui-Quadrado: {chi2_stat:.4f}")
print(f"Valor P: {p_val:.4f}")
print(f"Graus de Liberdade: {dof}")
print("Frequências Esperadas:\n", pd.DataFrame(ex, index=tabela.index, columns=tabela.columns))




################ ANOVA ##########################
# Vamos demonstrar um exemplo com o teste Anova
#H0: Não há diferença no valor médio gasto com bebidas em nenhuma das populações
#H1: Há diferença no valor médio gasto com bebidas em pelo menos uma das populações


# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:07:55 2024

@author: asgunzi
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dados
dados_anova = pd.DataFrame({
    'Gastos': [174.770021661909, 161.329206619394, 153.679900850863, 163.790338797433, 141.363480335882, 175.351592994046, 
               185.793398289321, 184.720273514352, 163.400459287948, 170.202462740626, 150.8549565713, 167.583106239899, 
               140.190492201897, 157.440088617225, 171.596654773339, 138.885665257324, 147.942698809323, 9.87474262516482, 
               50.5645554670016, 14.2586307887884, 8.5061846804934, 25.0875496696788, 17.0661987504312, 41.3867417301938, 
               20.8113941426179, 60.1224674502026, 35.5154028285664, 23.7622285692359, 34.6086119259266, 30.4321086925016, 
               27.8188980544904, 37.4729772794009, 30.7229538650678, 48.0452539322412, 78.9197865324734, 42.4926762466659, 
               8.81227865272712, 39.5751781629677, 37.1329656327517, 15.8016718071775, 5.74735216885902, 38.684069121093, 
               30.9398891106907, 34.7370783113952, 13.2630510987537, 19.6212096123791, 16.716945267481, 24.4037922212213, 
               4.63398786180773, 32.9436217626275, 21.511905851158, 31.4997283634204, 26.6610570873775, 34.6304034101472, 
               16.2704826042681, 11.2323425300881, 18.023244405391, 15.4790632095655, 8.25633422881043, 27.9053307974433, 
               72.3298402892867, 4.7263338963663, 14.4153129255327, 41.2234268777169, 50.5684226296565, 19.8344282661234, 
               8.81306901471397, 19.5112436004646, 55.6251926080436, 16.7592556127806, 20.3176176298076, 31.2073058210955, 
               17.0613250010048, 47.8590627884627, 2.59778754862417, 35.9470130480825, 2.39404093355522, 9.38425601777391, 
               25.2455048267186, 16.1960287769175, 43.530118783298, 32.7250288712979, 5.43268078364765, 44.5365791890593, 
               32.9831443965413, 28.2104605365607, 3.18609515001209, 14.3698142789208, 39.9617218607622, 50.564581262513, 
               10.4634451365926, 36.4842442182048, 13.1330189654278, 8.93702642184252, 12.1501174131844, 22.2552757873296, 
               15.1407470062459, 11.7525513477354, 16.2990775324815, 24.4627568806115, 2.87916580644454, 44.5453919973285, 
               38.0393535792355, 32.1985589022666, 0.357075783631849, 22.0703974352325, 50.7486034030794, 18.604230207709, 
               5.83122133978906, 19.9252025339318, 6.8366108202567, 27.5834177510951, 41.9303025963975, 3.077799353254, 
               28.0507001837521, 33.0042729903, 50.7366690908169, 30.1697285113061, 6.53184416916073, 7.53469171526227, 
               5.49225229796712, 9.53198727121377, 6.59266645551752, 19.8423174628847, 0.781567028951091, 22.1605754480815, 
               5.90830712162365, 54.3457453874529, 33.3341495203441, 37.2034845899045],
    'Estado_Civil': ["solteiros", "solteiros", "solteiros", "solteiros", 
                    "solteiros", "solteiros", "solteiros", "solteiros", "solteiros", 
                    "solteiros", "solteiros", "solteiros", "solteiros", "solteiros", 
                    "solteiros", "solteiros", "solteiros", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Casados", "Casados", "Casados", "Casados", "Casados", "Casados", 
                    "Divorciados", "Divorciados", "Divorciados", "Divorciados", "Divorciados", 
                    "Divorciados", "Divorciados", "Divorciados", "Divorciados", "Divorciados", 
                    "Divorciados", "Divorciados", "Divorciados", "Divorciados", "Divorciados"]
})

# Visualiza o conjunto de dados
print(dados_anova.head())

# Visualiza a variabilidade nas distintas populações com uso de boxplot
dados_anova.boxplot(column='Gastos', by='Estado_Civil')

# Agrupa os dados por Estado_Civil
grouped = dados_anova.groupby('Estado_Civil')['Gastos']

# Aplica a ANOVA
f_stat, p_val = stats.f_oneway(*[group for name, group in grouped])

# Resultados
print(f"Estatística F: {f_stat}")
print(f"Valor P: {p_val}")

# Decisão
if p_val < 0.05:
    print("Rejeitamos a hipótese nula (H0). Existem diferenças significativas entre os grupos.")
else:
    print("Não rejeitamos a hipótese nula (H0). Não há evidência suficiente para afirmar que existem diferenças significativas entre os grupos.")
















