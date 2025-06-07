import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
import base64
from io import BytesIO

# Datos 
np.random.seed(42)
fechas= pd.date_range('2023-01-01', '2024-12-31', freq='D') 
num_productos= ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']

# dataset
data = []

for fecha in fechas:
    for _ in range(np.random.poisson(10)):
        data.append({
            'fecha': fecha,
            'producto' : np.random.choice(num_productos),
            'region' : np.random.choice(regiones),
            'cantidad': np.random.randint(1, 6),
            'precio_unitario': np.random.uniform(50, 1500),
            'vendedor' : f'Vendedor_{np.random.randint(1, 21)}'
        })
        
df = pd.DataFrame(data)

"""df.head(10)

# columna de ventas totales
df['venta_total'] = df['cantidad'] * df['precio_unitario']
df.head()

# Shape. (filas,columnas)
print(f'Shape del dataset: \n {df.shape}\n')

# info. (tipo de datos, valores nulos, memoria)
print(f'Información general del dataset: \n {df.info()}\n')
df.info()

# estadisticas descriptivas
print(f'Estadísticas descriptivas: \n {df.describe()}\n')

# ventas por mes
df_mes = df.groupby(df['fecha'].dt.to_period('M'))['venta_total'].sum().reset_index()
df_mes['fecha'] = df_mes['fecha'].astype(str)
df_mes.head() 

# gr áfico de lineas
fig_mes = px.line(df_mes, x='fecha', y='venta_total',
                  title='Tendencia de Ventas Mensuales',
                  labels={'venta_total': 'Ventas (€)', 'Fecha': 'mes'})

fig_mes.update_traces(line=dict(width=3))
#fig_mes.show(renderer='browser')

# Top productos más vendidos
df_productos = df.groupby('producto')['venta_total'].sum().sort_values(ascending=True)
fig_products = px.bar(x=df_productos.values, y=df_productos.index,
                      orientation='h',title='Ventas por Producto',
                      labels={'x': 'Ventas Totales(€)', 'y': 'Producto'})
#fig_products.show(renderer='browser')

# Análisis por región
df_region = df.groupby('region')['venta_total'].sum().reset_index()
fig_region = px.pie(df_region, values='venta_total', names='region',
                    title='Distribución de Ventas por Región',
                    labels={'venta_total': 'Ventas Totales(€)', 'region': 'Región'})    

#fig_region.show(renderer='browser')

# correlacion entre variables
df_coor = df[['cantidad', 'precio_unitario', 'venta_total']].corr()

fig_heatmap = px.imshow(df_coor, text_auto=True, aspect='auto',
                        title='Correlación entre Variables Numéricas')
#fig_heatmap.show(renderer='browser')

# Distribución de ventas 
fig_histograma = px.histogram(df, x='venta_total', nbins=20,
                              title='Distribución de Ventas Totales')
#fig_histograma.show(renderer='browser')
"""
# configuracion del dashboard
st.set_page_config(page_title='Dashboard de Ventas',
                   page_icon='logo.ico', layout='wide')

def get_image_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return img_base64
    except FileNotFoundError:
        st.error('No se pudo cargar la imagen del logo. ❓')
        return None

# Obtener la imagen en base64
img_base64 = get_image_base64("Logo.png")

# Insertar en el título
if img_base64:
    st.markdown(
        f"""
        <h1 style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{img_base64}" width="40" style="margin-right: 10px;" />
            Dashboard de Ventas
        </h1>
        """,
        unsafe_allow_html=True
    )

st.markdown('---')

# Slider para filtros
st.sidebar.header('Filtros')
productos_seleccionados = st.sidebar.multiselect(
    'Selecciona Productos:',
    options=df['producto'].unique(),
    default=df['producto'].unique()
)

# slider para regiones
regiones_seleccionadas = st.sidebar.multiselect(
    'Selecciona Regiones:',
    options=df['region'].unique(),
    default=df['region'].unique()
)

# filtrar datos basado en selecciones
df_filtrado = df[
    (df['producto'].isin(productos_seleccionados)) & 
    ( df['region'].isin(regiones_seleccionadas))
]



# Métricas Principales
col1, col2, col3, col4 = st.columns(4) 
with col1:
    st.metric(label='Total de Ventas',
              value=f'{df_filtrado["venta_total"].sum():,.0f} €'
             )
with col2:
    st.metric(label='Promedio de Ventas',
              value=f'{df_filtrado["venta_total"].mean():,.0f} €'
             )
with col3:
    st.metric("Número de Ventas",f'{len(df_filtrado)}'
             )
with col4:
    crecimiento = ((df_filtrado[df_filtrado['fecha'] >= '2024-01-01']['venta_total'].sum())/
                   (df_filtrado[df_filtrado['fecha'] < '2024-01-01']['venta_total'].sum())-1)*100
    st.metric('Crecimiento de Ventas 2024',
             f'{crecimiento:.2f}%'
             )
# Layout con dos colummnas
col1,col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_mes, use_container_width=True)
    st.plotly_chart(fig_products, use_container_width=True)

with col2:
    st.plotly_chart(fig_region, use_container_width=True)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Gráfico completo en la parte inferior
st.plotly_chart(fig_histograma, use_container_width=True)


