#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[13]:


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
df.head(10)


# In[14]:


# columna de ventas totales
df['venta_total'] = df['cantidad'] * df['precio_unitario']
df.head()


# EXPLORANDO DATOS

# In[15]:


# Shape. (filas,columnas)
print(f'Shape del dataset: \n {df.shape}\n')


# In[16]:


# info. (tipo de datos, valores nulos, memoria)
print(f'Información general del dataset: \n {df.info()}\n')
df.info()


# In[17]:


# estadisticas descriptivas
print(f'Estadísticas descriptivas: \n {df.describe()}\n')


# In[18]:


# ventas por mes
df_mes = df.groupby(df['fecha'].dt.to_period('M'))['venta_total'].sum().reset_index()
df_mes['fecha'] = df_mes['fecha'].astype(str)
df_mes 


# In[ ]:


# gráfico de lineas
fig_mes = px.line(df_mes, x='fecha', y='venta_total',
                  title='Tendencia de Ventas Mensuales',
                  labels={'venta_total': 'Ventas (€)', 'fecha': 'Mes'})

fig_mes.update_traces(line=dict(width=3))
#fig_mes.show()


# In[ ]:


# Top productos más vendidos
df_productos = df.groupby('producto')['venta_total'].sum().sort_values(ascending=True)
fig_products = px.bar(x=df_productos.values, y=df_productos.index,
                      orientation='h',title='Ventas por Producto',
                      labels={'x': 'Ventas Totales(€)', 'y': 'Producto'})
#fig_products.show()


# In[ ]:


# Análisis por región
df_region = df.groupby('region')['venta_total'].sum().reset_index()
fig_region = px.pie(df_region, values='venta_total', names='region',
                    title='Distribución de Ventas por Región',
                    labels={'venta_total': 'Ventas Totales(€)', 'region': 'Región'})    

#fig_region.show()


# In[ ]:


# correlacion entre variables
df_coor = df[['cantidad', 'precio_unitario', 'venta_total']].corr()

fig_heatmap = px.imshow(df_coor, text_auto=True, aspect='auto',
                        title='Correlación entre Variables Numéricas')
#fig_heatmap.show()


# In[ ]:


# Distribución de ventas 
fig_histograma = px.histogram(df, x='venta_total', nbins=20,
                              title='Distribución de Ventas Totales')
#fig_histograma.show()

#fig_histograma.show(renderer='browser')


# CONFIGURACIÓN DEl DASHBOARD

# In[ ]:


# Titulo e icono
st.set_page_config(page_title='Dashboard de Ventas',
                   page_icon='📊', layout='wide')


# In[ ]:


# Título Principal
st.title('📊 Dashboard de Ventas')
st.markdown('---')


# In[35]:


# Slider para filtros
st.sidebar.header('Filtros')
productos_seleccionados = st.sidebar.multiselect(
    'Selecciona Productos:',
    options=df['producto'].unique(),
    default=df['producto'].unique()
)


# In[ ]:




