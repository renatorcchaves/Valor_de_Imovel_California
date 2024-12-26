# Modelo de Regressão para Estimar Valor de Imóvel da California 

**Origem da base de dados:** https://www.kaggle.com/datasets/camnugent/california-housing-prices/data

**Contexto**
Este conjunto de dados foi derivado do censo dos EUA de 1990, em que cada linha do dataframe que será gerado representa um grupo de blocos censitários. Um grupo de blocos é a menor unidade geográfica para a qual o
Escritório do Censo dos EUA publica dados amostrais (um grupo de blocos geralmente tem  uma população de 600 a 3.000 pessoas e é composto por vários domicílios). Por conta disso, para cada linha do dataframe encontraremos grandes números de cômodos, quartos, e população. Em alguns casos, esses números serão surpreendentemente altos (destoando dos valores médios), pois dentro de um bloco censitário podem conter estabelecimentos como hotéis / resort de férias. 

A variável alvo a ser predita é o valor mediano das casas para os distritos da Califórnia, expressa em dólares.

**Features da base de dados**
- `median_income`: renda mediana no grupo de blocos (em dezenas de milhares de dólares)
- `housing_median_ag`e: idade mediana das casas no grupo de blocos
- `total_rooms`: número cômodos no grupo de blocos
- `total_bedrooms`: número de quartos no grupo de blocos
- `population`: população do grupo de blocos
- `households`: domicílios no grupo de blocos
- `latitude`: latitude do grupo de blocos
- `longitude`: longitude do grupo de blocos
- `ocean_proximity`: proximidade do oceano
    - `NEAR BAY`: perto da baía
    - `<1H OCEAN`: a menos de uma hora do oceano
    - `INLAND`: no interior
    - `NEAR OCEAN`: perto do oceano
    - `ISLAND`: ilha
- `median_house_value`: valor mediano das casas no grupo de blocos

Abaixo temos uma demonstração de como o valor mediano dos domícilios podem variar para cada condado, e como os domicílios estão espalhados pelos condados

![Valor Mediano dos Domicilios por Condado](relatorios/Valor%20mediano%20dos%20domicilios.png)     
![Distribuição dos blocos censitários pelos condados](relatorios/Domicilios%20por%20condado.png)

## **Etapas na construção do modelo de regressão**
1) Análise Exploratória dos Dados
2) Avaliar os dados geográficos com Geopandas
3) Desenvolver notebook "padrão" para o projeto de Machine Learning, com funções que podem facilitar testes de diferentes modelos
4) Aprimorar o modelo de Machine Learning, seguindo as etapas abaixo:
    - Linear Regression com preprocessamento das colunas categóricas (*One_hot_encoder* + *Ordinal_Encoder*).
    - Avaliar se a transformação de target melhora modelo, e testar qual transformação é melhor (*PowerTransform* ou *Quantile Transform*)
    - Testar diferentes tipos de preprocessamento para colunas numéricas (somente *RobustScaler* ou *Standard Scaler* + *Power Transform*) com a melhor transformação de target
    - Aplicar *Polynomial Features* para verificar qual grau polinomial gera o melhor modelo
    - Substituir o *Linear Regression* pelo *Elastic Net* para verificar se a regularização dos dados melhora o modelo
    - Substituir o *ElasticNet* pelo regressor *Ridge*, testando os melhores parâmetros do modelo Ridge, e verificar os resultado para finalmente obter melhor modelo de regressão para esse projeto

Para chegar no melhor modelo, foi realizado testes com diferentes parâmetros de preprocessamento / transformação de target / modelos de regressão através da ferramenta GridSearchCV do Scikit-learn. Em cada etapa desses testes, foi aplicado validações cruzadas para avaliar se as métricas de erro como R², RMSE e MAE (Root Mean Squared Erros e Mean Absolute Error) indicavam que o modelo gerou melhores resultados. 

Por fim, através do modelo final, foi comparado os resultados preditos pelo modelo com os resultados da base de teste, como é mostrado na imagem abaixo:

![Valores Preditos X Valores Reais](relatorios/Analisando%20Valores%20Preditos%20X%20Valores%20Reais%20-%20Melhor%20Modelo.png)

## Organização das pastas do projeto

```
├── .env               <- Arquivo de variáveis de ambiente (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── ambiente.yml       <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto se uma for escolhida
├── README.md          <- README principal para desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── modelos            <- Modelos treinado e serializado para predizer a variável alvo
|
├── notebooks          <- Arquivos tipo .ipynb desenvolvidos para análise dos dados e geração do modelo de regressão
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py  <- Torna um módulo Python
|      ├── config.py    <- Configurações básicas do projeto
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados
|
├── relatorios         <- Análises geradas durante o projeto, como arquivos de imagem, PDF, etc.
