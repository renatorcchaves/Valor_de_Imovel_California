# ----- IMPORTANDO BIBLIOTECAS ----------------------------------------------------------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pydeck as pdk

from joblib import load

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

# ----- FUNÇÕES PARA CARREGAR OS DADOS ----------------------------------------------------------------------------------------------------------------------------

# Criando funções para carregar os arquivos que serão usados pelo streamlit (elas serão usadas para que o streamlit não fique recarregando os dados a cada interação na plataforma)
# Colocar um @ antes de uma função permite ao streamlit rodar uma outra função antes da função abaixo. No caso, é uma função de cache
# Essa função de cachê pré-armazena algumas informações dessa base de dados para rodá-la mais rapidamente num segundo momento. É um "decorador" da base de dados
# @st.cache_data para bases de dados, e st.cache_resource para modelos de machine learning e demais aplicativos

# @st.cache_resource                    
def carregar_modelo():
    return load(MODELO_FINAL)

@st.cache_data                        
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

# OBS: O carregamento dos dados gdf_geo é bem mais completo porque a coluna 'geometry' não estava sendo renderizada no mapa do pydeck de acordo com o condado selecionado                
@st.cache_data
def carregar_dados_geo():                  
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Functions to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)     # Fix invalid geometry
        #Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo['geometry'] = gdf_geo['geometry'].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x,y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x,y] for x, y, in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Apply the coordinate conversion and store in a new column
    gdf_geo['geometry'] = gdf_geo['geometry'].apply(get_polygon_coordinates)

    return gdf_geo

df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# ----- DESENVOLVIMENTO DE PÁGINA DO STREAMLIT ----------------------------------------------------------------------------------------------------------------------------

st.title("Previsão de preço de imóveis")

condados = sorted(gdf_geo['name'].unique())

coluna1, coluna2 = st.columns(2)              # Dividindo a página do Streamlit em 2 colunas:

with coluna1:                                 # with .... é um gerenciador de contexto
    with st.form(key='formulario'):
    
        # Adicionando os Input's widgets
        selecionar_condado = st.selectbox("Condado", condados)
        housing_median_age = st.number_input('Idade do imóvel', value=10, min_value=1, max_value=50)
        median_income = st.slider('Renda média (milhares de US$)', min_value = 5.0, max_value = 100.0, value=45.0, step = 5.0)   # O max_value segue o valor máximo dessa variável no gdf_geo

        # Devemos deixar widget's input pro usuário determinar todas as variáveis? São valores para um conjunto de imóveis. O usuário pode não saber que valor usar dependendo do caso
        # Podemos deixar com um valor padrão por baixo dos panos que faça sentido pro modelo? 
        # Usaremos o DADOS_GEO_MEDIAN para ver os valores MEDIANO das variáveis abaixo de acordo com o condado
        longitude = gdf_geo.query("name == @selecionar_condado")['longitude'].values                 #llongitude = st.number_input('Longitude', value=-122.33)
        latitude = gdf_geo[gdf_geo["name"] == selecionar_condado]["latitude"].values                 #latitude = st.number_input('Latitude', value=37.88)
        total_rooms = gdf_geo.query("name == @selecionar_condado")['total_rooms'].values             #total_rooms = st.number_input('Total de cômodos', value=800)
        total_bedrooms = gdf_geo.query("name == @selecionar_condado")['total_bedrooms'].values       #total_bedrooms = st.number_input('Total de quartos', value=100)
        population = gdf_geo[gdf_geo["name"] == selecionar_condado]["population"].values             #population = st.number_input('População', value=300)
        households = gdf_geo[gdf_geo["name"] == selecionar_condado]["households"].values             #households = st.number_input('Domicílios', value=100)
        ocean_proximity = gdf_geo.query("name == @selecionar_condado")['ocean_proximity'].values     #ocean_proximity = st.selectbox('Proximidade do oceano', df['ocean_proximity'].unique())
        rooms_per_household = gdf_geo.query("name == @selecionar_condado")['rooms_per_household'].values
        bedrooms_per_room = gdf_geo.query("name == @selecionar_condado")['bedrooms_per_room'].values
        population_per_household = gdf_geo.query("name == @selecionar_condado")['population_per_household'].values
        #rooms_per_household = (total_rooms / households)                                             #rooms_per_household = st.number_input('Cômodos por domícilio', value = 7)
        #bedrooms_per_room = (total_bedrooms / total_rooms)                                           #bedrooms_per_room = st.number_input('Quartos por cômodo', value = 0.2)
        #population_per_household = (population / households)                                         #population_per_household = st.number_input('Pessoas por domícilio', value = 2)

        # Variáveis calculadas a partir das outras variáveis definidas acima (optei por usar a mediana dessas variáveis pelo gdf_geo)
        bins_income_cat = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(median_income / 10, bins=bins_income_cat)                         #median_income_cat = st.number_input('Categoria de renda', value = 4)
        # Essa função do numpu np.digitize recebe um número (median_income) e determina a qual categoria pertence esse número de acordo com os as divisões (bins) que ela receber

        entrada_modelo = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income / 10,
        'ocean_proximity': ocean_proximity,
        'median_income_cat': median_income_cat,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
        }

        df_entrada_modelo = pd.DataFrame(entrada_modelo)#, index=[0])        # index=[0] --> É só porque deu um erro no streamlit devido falta de índice 
        
        botao_previsao = st.form_submit_button("Prever preço")

    if botao_previsao:                                 # Se a pessoa tiver clicado no botão, essa variável será "Verdadeira"
        try:
            preco = modelo.predict(df_entrada_modelo)
            st.metric(label="Preço previsto:", value=f"US$ {preco[0][0]:.2f}")

        except Exception as e:
            st.error(f'Ocorreu um erro durante a previsao: {e}')

with coluna2:
    # Estado de visualização inicial
    view_state = pdk.ViewState(
        latitude = float(latitude[0]),            # Precisa fornecer o float() antes da variáveis pois estava dando um erro no streamlit
        longitude = float(longitude[0]),          # Algumas bibliotecas não funcionam muito bem com dados float32, colocando o float(variável) transformamos ela em float64
        zoom = 5,
        min_zoom = 5,
        max_zoom= 15
    )

    # Adicionando uma camada para destacar o estado da California
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data = gdf_geo[['geometry', 'name']],
        get_polygon = 'geometry',
        get_fill_color = [0, 0, 255, 100],             # Seguindo o sistema de cores RGB (digitar no wikipedia) - a cor azul é identificada como 0, 0, 255 e o branco 255, 255, 255
        get_line_color = [255, 255, 255],
        get_line_width=50,
        pickable = True,
        auto_highlight = True
    )

    # Adicionando uma camada para destacar o condado selecionado no streamlit
    condado_selecionado = gdf_geo.query('name == @selecionar_condado')
    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data = condado_selecionado[['name', 'geometry']],
        get_polygon = 'geometry',
        get_fill_color = [255, 0, 0, 100],             # Cor vermelha
        get_line_color = [0, 0, 0],                    # Cor preta
        get_line_width=500,
        pickable = True,
        auto_highlight = True
    )

    # Adicionando uma 'tooltip' (dica de ferramenta) pra destacar o nome do condado quando passamos o mouse por cima
    tooltip = {
        "html": "<b>Condado:</b> {name}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontsize": "10px"}
    }

    # Mapa do PyDeck
    mapa = pdk.Deck(
        initial_view_state = view_state,
        map_style='light', 
        layers = [polygon_layer, highlight_layer],
        tooltip = tooltip
    )

    # Adicionando o mapa ao streamlit
    st.pydeck_chart(mapa)