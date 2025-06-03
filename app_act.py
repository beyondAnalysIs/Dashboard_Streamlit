import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots # No se usa actualmente
# import seaborn as sns # No se usa actualmente
# import matplotlib.pyplot as plt # No se usa actualmente
from datetime import datetime, timedelta

# --- Configuración de la Página y Título Principal ---
st.set_page_config(page_title='Dashboard de Ventas Interactivo',
                   page_icon='📊', layout='wide')

st.title('📊 Dashboard de Ventas Interactivo')
st.markdown('---')

# --- Generación de Datos de Ejemplo (igual que en tu código original) ---
@st.cache_data # Usar st.cache_data para optimizar la carga de datos
def cargar_datos():
    np.random.seed(42)
    fechas = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    num_productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
    regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    data = []
    for fecha in fechas:
        for _ in range(np.random.poisson(10)): # Generar un número aleatorio de ventas por día
            data.append({
                'fecha': fecha,
                'producto': np.random.choice(num_productos),
                'region': np.random.choice(regiones),
                'cantidad': np.random.randint(1, 6),
                'precio_unitario': np.random.uniform(50, 1500),
                'vendedor': f'Vendedor_{np.random.randint(1, 21)}'
            })
    df = pd.DataFrame(data)
    df['venta_total'] = df['cantidad'] * df['precio_unitario']
    # Convertir fecha a datetime si no lo está ya (aunque date_range lo hace)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

df_original = cargar_datos()

# --- Barra Lateral de Filtros ---
st.sidebar.header('Filtros')

# Filtro de Productos
productos_disponibles = sorted(df_original['producto'].unique())
productos_seleccionados = st.sidebar.multiselect(
    'Selecciona Productos:',
    options=productos_disponibles,
    default=productos_disponibles
)

# Filtro de Regiones
regiones_disponibles = sorted(df_original['region'].unique())
regiones_seleccionadas = st.sidebar.multiselect(
    'Selecciona Regiones:',
    options=regiones_disponibles,
    default=regiones_disponibles
)

# Filtro de Rango de Fechas
min_fecha = df_original['fecha'].min().date()
max_fecha = df_original['fecha'].max().date()

fecha_inicio, fecha_fin = st.sidebar.date_input(
    "Selecciona Rango de Fechas:",
    value=(min_fecha, max_fecha),
    min_value=min_fecha,
    max_value=max_fecha,
    key='date_range_picker' # Añadir una clave única si hay múltiples date_input
)

fecha_inicio=pd.to_datetime(fecha_inicio)
fecha_fin=pd.to_datetime(fecha_fin)

# Asegurarse de que fecha_inicio y fecha_fin son Timestamps para la comparación
if isinstance(fecha_inicio, datetime):
    fecha_inicio = pd.Timestamp(fecha_inicio)
if isinstance(fecha_fin, datetime):
    fecha_fin = pd.Timestamp(fecha_fin)


# --- Filtrar Datos Basado en Selecciones ---
# Asegurarse de que las fechas de inicio y fin no sean None
if fecha_inicio is not None and fecha_fin is not None:
    df_filtrado = df_original[
        (df_original['producto'].isin(productos_seleccionados)) &
        (df_original['region'].isin(regiones_seleccionadas)) &
        (df_original['fecha'] >= fecha_inicio) &
        (df_original['fecha'] <= fecha_fin)
    ]
else:
    # Si las fechas no están seleccionadas, usar todos los datos (o manejar como prefieras)
    df_filtrado = df_original[
        (df_original['producto'].isin(productos_seleccionados)) &
        (df_original['region'].isin(regiones_seleccionadas)) &
        (df_original['fecha'] >= fecha_inicio) &
        (df_original['fecha'] <= fecha_fin)
    ]


# --- Mostrar un mensaje si no hay datos después de filtrar ---
if df_filtrado.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
else:
    # --- Métricas Principales (usando df_filtrado) ---
    st.subheader("Métricas Principales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='Total de Ventas',
                  value=f'{df_filtrado["venta_total"].sum():,.0f} €')
    with col2:
        st.metric(label='Promedio de Ventas',
                  value=f'{df_filtrado["venta_total"].mean():,.0f} €')
    with col3:
        st.metric("Número de Ventas", f'{len(df_filtrado):,}')

    with col4:
        # Cálculo de crecimiento robusto
        ventas_2023_filtradas = df_filtrado[df_filtrado['fecha'].dt.year == 2023]['venta_total'].sum()
        ventas_2024_filtradas = df_filtrado[df_filtrado['fecha'].dt.year == 2024]['venta_total'].sum()

        if ventas_2023_filtradas > 0:
            crecimiento = ((ventas_2024_filtradas / ventas_2023_filtradas) - 1) * 100
            st.metric('Crecimiento Ventas (2024 vs 2023)', f'{crecimiento:.2f}%')
        elif ventas_2024_filtradas > 0:
             st.metric('Crecimiento Ventas (2024 vs 2023)', 'N/A (No hay ventas en 2023)')
        else:
            st.metric('Crecimiento Ventas (2024 vs 2023)', '0.00% (No hay ventas)')
            
    st.markdown('---')
    st.subheader("Visualizaciones Interactivas")

    # --- Generación de Gráficos (usando df_filtrado) ---

    # 1. Tendencia de Ventas Mensuales
    df_mes_filtrado = df_filtrado.copy() # Copiar para evitar SettingWithCopyWarning
    df_mes_filtrado['fecha_mes'] = df_mes_filtrado['fecha'].dt.to_period('M')
    df_mes_plot = df_mes_filtrado.groupby('fecha_mes')['venta_total'].sum().reset_index()
    df_mes_plot['fecha_mes'] = df_mes_plot['fecha_mes'].astype(str) # Convertir Period a string para Plotly

    fig_mes = px.line(df_mes_plot, x='fecha_mes', y='venta_total',
                      title='Tendencia de Ventas Mensuales',
                      labels={'venta_total': 'Ventas (€)', 'fecha_mes': 'Mes'},
                      markers=True)
    fig_mes.update_traces(line=dict(width=3))

    # 2. Top Productos Más Vendidos
    df_productos_plot = df_filtrado.groupby('producto')['venta_total'].sum().sort_values(ascending=False).reset_index()
    fig_products = px.bar(df_productos_plot,
                          x='venta_total', y='producto',
                          orientation='h', title='Ventas por Producto',
                          labels={'venta_total': 'Ventas Totales (€)', 'producto': 'Producto'},
                          color='producto')
    fig_products.update_layout(yaxis={'categoryorder':'total ascending'})


    # 3. Distribución de Ventas por Región
    df_region_plot = df_filtrado.groupby('region')['venta_total'].sum().reset_index()
    fig_region = px.pie(df_region_plot, values='venta_total', names='region',
                        title='Distribución de Ventas por Región',
                        labels={'venta_total': 'Ventas Totales (€)', 'region': 'Región'},
                        hole=0.3)
    fig_region.update_traces(textposition='inside', textinfo='percent+label')


    # 4. Correlación entre Variables Numéricas
    # Seleccionar solo columnas numéricas para la correlación
    df_corr_seleccion = df_filtrado[['cantidad', 'precio_unitario', 'venta_total']]
    if not df_corr_seleccion.empty and len(df_corr_seleccion) > 1: # Se necesita al menos 2 filas para calcular correlación
        df_coor_plot = df_corr_seleccion.corr()
        fig_heatmap = px.imshow(df_coor_plot, text_auto=True, aspect='auto',
                                title='Correlación entre Variables Numéricas',
                                color_continuous_scale='RdBu_r', # Escala de color más común para correlaciones
                                labels=dict(color="Coeficiente de Correlación"))
    else:
        fig_heatmap = go.Figure().update_layout(title='Correlación (datos insuficientes)')


    # 5. Distribución de Ventas (Histograma)
    fig_histograma = px.histogram(df_filtrado, x='venta_total', nbins=30, # Aumentar nbins para más detalle si es necesario
                                  title='Distribución de Ventas Totales',
                                  labels={'venta_total': 'Venta Total (€)'},
                                  marginal="box") # Añadir un box plot marginal

    # --- Layout del Dashboard ---
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.plotly_chart(fig_mes, use_container_width=True)
        st.plotly_chart(fig_products, use_container_width=True)

    with col_graf2:
        st.plotly_chart(fig_region, use_container_width=True)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.plotly_chart(fig_histograma, use_container_width=True)

    # --- Mostrar Tabla de Datos Filtrados (Opcional) ---
    if st.checkbox("Mostrar datos filtrados"):
        st.subheader("Datos Filtrados")
        # Mostrar menos columnas para mejor visualización o permitir selección
        st.dataframe(df_filtrado[['fecha', 'producto', 'region', 'cantidad', 'precio_unitario', 'venta_total', 'vendedor']].head(100))
