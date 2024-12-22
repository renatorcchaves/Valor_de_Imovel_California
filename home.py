import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd

from joblib import load

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

# Criando funções para carregar os arquivos que serão usados pelo streamlit (elas serão usadas para que o streamlit não fique recarregando os dados a cada interação na plataforma)

@st.cache_data                        # Colocar um @ antes de uma função permite ao streamlit rodar uma outra função antes da função abaixo. No caso, é uma função de cache
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data                        # Essa função de cachê pré-armazena algumas informações dessa base de dados para rodá-la mais rapidamente num segundo momento. É um "decorador" da base de dados
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)

@st.cache_resource                    # @st.cache_data para bases de dados, e st.cache_resource para modelos de machine learning e demais aplicativos
def carregar_modelo():
    return load(MODELO_FINAL)

df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

st.title("Previsão de preço de imóveis")

# Adicionando os Input's widgets
longitude = st.number_input('Longitude', value=-122.33)
latitude = st.number_input('Latitude', value=37.88)
housing_median_age = st.number_input('Idade do imóvel', value=10)
total_rooms = st.number_input('Total de cômodos', value=800)
total_bedrooms = st.number_input('Total de quartos', value=100)
population = st.number_input('População', value=300)
households = st.number_input('Domicílios', value=100)
median_income = st.slider('Renda média (múltiplos de  US$ 10k)', min_value = 0.15, max_value = 15.0, value=4.5, step = 0.5)
ocean_proximity = st.selectbox('Proximidade do oceano', df['ocean_proximity'].unique())
median_income_cat = st.number_input('Categoria de renda', value = 4)
rooms_per_household = st.number_input('Cômodos por domícilio', value = 7)
bedrooms_per_room = st.number_input('Quartos por cômodo', value = 0.2)
population_per_household = st.number_input('Pessoas por domícilio', value = 2)

# Entradas do modelo de acordo com os valores inputados pelos widgets

botao_previsao = st.button("Prever preço")

entrada_modelo = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_proximity,
    'median_income_cat': median_income_cat,
    'rooms_per_household': rooms_per_household,
    'bedrooms_per_room': bedrooms_per_room,
    'population_per_household': population_per_household,
}

df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0])        # index=[0] --> É só porque deu um erro no streamlit devido falta de índice 

if botao_previsao:                                 # Se a pessoa tiver clicado no botão, essa variável será "Verdadeira"
    try:
        preco = modelo.predict(df_entrada_modelo)
        st.write(f'Preço previsto: $ {preco[0][0]:.2f}')

    except Exception as e:
        st.error(f'Ocorreu um erro durante a previsao: {e}')