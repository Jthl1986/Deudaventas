import pandas as pd
import streamlit as st
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="Deuda - Ventas", page_icon="📈")

hide_streamlit_style = """
    <style>
    /* Ocultar el botón de Share */
    .stApp header [data-testid="stHeader"] { 
        display: none; 
    }
    /* Opcional: Ocultar la barra de menú de Streamlit (incluye acceso al GitHub si está visible) */
    footer {visibility: hidden;}
    .css-1lsmgbg.egzxvld1 {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

url = "https://raw.githubusercontent.com/Jthl1986/T1/main/iipcOct25.csv"
#iipcJul25.csv
df = pd.read_csv(url, encoding='ISO-8859-1', sep=',')

hide_github_link = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_github_link, unsafe_allow_html=True)

def col_letter_to_index(letter):
    """Convertir una letra de columna de Excel a un índice de columna de Pandas (0-based)."""
    index = 0
    for char in letter.upper():
        index = index * 26 + (ord(char) - ord('A')) + 1
    return index - 1

def graficobar(df_final):
    def millions(x, pos):
        return '%1.0fM' % (x * 1e-6)
    
    # Verificar datos de entrada
    if df_final.empty:
        st.warning("DataFrame vacío: no hay datos para graficar.")
        return
    
    try:
        # Convertir fechas con manejo de errores
        df_final['fecha'] = pd.to_datetime(df_final['fecha'], format='%m/%Y', errors='coerce')
        df_final = df_final.dropna(subset=['fecha'])
        
        # Mapear meses a español
        month_translation = {
            'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
            'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
            'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
            'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
        }
        df_final['Mes'] = df_final['fecha'].dt.strftime('%B').map(month_translation)
        df_final['Año'] = df_final['fecha'].dt.year

        # Ordenar meses en español
        meses_ordenados = [
            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
        ]
        df_final['Mes'] = pd.Categorical(df_final['Mes'], categories=meses_ordenados, ordered=True)
        
        # Crear pivot y validar
        df_pivot = df_final.pivot_table(index='Mes', columns='Año', values='Ventas', aggfunc='sum')
        df_pivot = df_pivot.sort_index(axis=1)
        
        #st.write("Debug - Años disponibles:", df_pivot.columns.tolist())  # Debug
        
        if len(df_pivot.columns) < 2:
            st.warning(f"Requiere al menos 2 años. Años recibidos: {df_pivot.columns.tolist()}")
            return
            
        # Filtrar meses con al menos 1 año de datos
        df_comparison = df_pivot.dropna(how='all')
        
        if df_comparison.empty:
            st.warning("No hay meses con datos comunes entre los años.")
            return

        # Graficar
        fig, ax = plt.subplots(figsize=(12, 8))
        df_comparison.plot(kind='bar', ax=ax)
        ax.set_title(f'Comparación de Ventas por Mes ({", ".join(map(str, df_comparison.columns))})')
        ax.set_xlabel('Mes')
        ax.set_ylabel('Ventas')
        ax.yaxis.set_major_formatter(FuncFormatter(millions))
        ax.legend(title='Año', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Limpiar memoria de matplotlib
        
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")

def grafico(df_final):
    def millions(x, pos):
        return '%1.0fM' % (x * 1e-6)
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Graficar la deuda como barras
    barplot = sns.barplot(x='Mes', y='Deuda', data=df_final, palette='Greens_d', ax=ax1, edgecolor='black', alpha=0.8)
    for bar in barplot.patches:
        bar.set_edgecolor('black')
    
    # Graficar las ventas (los NaN no se grafican)
    ax1.plot(df_final['Mes'], df_final['Ventas'], color='r', marker='o', markersize=10, linewidth=2.5, linestyle='-', drawstyle='default')
    
    ax1.set_ylabel('Deuda y Ventas (en millones)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticklabels(df_final['Mes'], rotation=90)
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))
    ax1.set_ylim(0, max(df_final['Deuda'].max(), df_final['Ventas'].max(skipna=True)) * 1.1)
    
    plt.title('Deuda y Ventas por Mes a valores constantes', fontsize=16)
    
    # Leyenda personalizada
    line = plt.Line2D([0], [0], color='r', lw=2.5, marker='o', markersize=10)
    bar = plt.Rectangle((0,0),1,1,fc="green", edgecolor='black')
    ax1.legend([line, bar], ['Ventas', 'Deuda'], loc='upper left')
    
    sns.despine(left=True, bottom=True)
    return st.pyplot(fig)

def graficomm(df_final):
    def millions(x, pos):
        return '%1.0fM' % (x * 1e-6)
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    barplot = sns.barplot(x='Mes', y='Deuda', data=df_final, palette='Greens_d', ax=ax1, edgecolor='black', alpha=0.8)
    for bar in barplot.patches:
        bar.set_edgecolor('black')
    
    # Calcular media móvil de las ventas (SMA3)
    #df_final['Ventas_suavizadas'] = df_final['Ventas'].rolling(window=3).mean()
    df_final['Ventas_suavizadas'] = df_final['Ventas'].rolling(window=3, min_periods=1).mean()
    ax1.plot(df_final['Mes'], df_final['Ventas_suavizadas'], color='r', marker='o', markersize=10, linewidth=2.5, linestyle='-', drawstyle='default')
    
    ax1.set_ylabel('Deuda y Ventas (en millones)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticklabels(df_final['Mes'], rotation=90)
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))
    ax1.set_ylim(0, max(df_final['Deuda'].max(), df_final['Ventas'].max(skipna=True)) * 1.1)
    
    plt.title('Deuda y Ventas en $ (Media Movil SMA3)', fontsize=16)
    
    line = plt.Line2D([0], [0], color='r', lw=2.5, marker='o', markersize=10)
    bar = plt.Rectangle((0,0),1,1,fc="green", edgecolor='black')
    ax1.legend([line, bar], ['Ventas', 'Deuda'], loc='upper left')
    
    sns.despine(left=True, bottom=True)
    return st.pyplot(fig)

def graficodol(df_final):
    def millions(x, pos):
        return '%1.0fM' % (x * 1e-6)
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    barplot = sns.barplot(x='Mes', y='Dolar', data=df_final, palette='Greens_d', ax=ax1, edgecolor='black', alpha=0.8)
    for bar in barplot.patches:
        bar.set_edgecolor('black')
    
    df_final['Ventas_suavizadasdol'] = df_final['dolares'].rolling(window=3, min_periods=1).mean()
    ax1.plot(df_final['Mes'], df_final['Ventas_suavizadasdol'], color='r', marker='o', markersize=10, linewidth=2.5, linestyle='-', drawstyle='default')
    
    ax1.set_ylabel('Deuda y Ventas (en millones)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticklabels(df_final['Mes'], rotation=90)
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))
    ax1.set_ylim(0, max(df_final['Dolar'].max(), df_final['dolares'].max(skipna=True)) * 1.1)
    
    plt.title('Deuda y Ventas en u$s (Media Movil SMA3)', fontsize=16)
    
    line = plt.Line2D([0], [0], color='r', lw=2.5, marker='o', markersize=10)
    bar = plt.Rectangle((0,0),1,1,fc="green", edgecolor='black')
    ax1.legend([line, bar], ['Venta en u$s', 'Deuda en u$s'], loc='upper left')
    
    sns.despine(left=True, bottom=True)
    return st.pyplot(fig)

def process_file(file):
    df_excel = pd.read_excel(file, header=None, engine='xlrd')
    celdas_totales = ['E4', 'K4', 'Q4', 'W4', 'AC4', 'AI4', 'AO4', 'AU4', 'BA4', 'BG4', 'BM4', 'BS4', 'BY4', 'CE4', 'CK4', 'CQ4', 'CW4', 'DC4', 'DI4', 'DO4', 'DU4', 'EA4', 'EG4', 'EM4']
    celdas_indices = [(int(''.join(filter(str.isdigit, cell))) - 1, col_letter_to_index(''.join(filter(str.isalpha, cell)))) for cell in celdas_totales]
    totales = [df_excel.iat[row, col] if pd.notna(df_excel.iat[row, col]) else 0 for row, col in celdas_indices]
    meses = df_excel.iloc[1, 3:].dropna().values
    min_length = min(len(meses), len(totales))
    meses = meses[:min_length]
    totales = totales[:min_length]
    df_result = pd.DataFrame({'Mes': meses, 'Nosis': totales})
    df_merged = df.merge(df_result, on='Mes', how='inner')
    df_merged['Nosis'] = df_merged['Nosis'].astype(float)
    df_merged['Deuda'] = (df_merged['iipc'] * df_merged['Nosis']).round(0).astype(float)
    df_merged['Dolar'] = (df_merged['Nosis'] / df_merged['dolar']).round(0).astype(float)
    return df_merged

def process_text_input(text_input, sistema_viejo=True):
    data = StringIO(text_input)
    df_text = pd.read_csv(data, sep='\t')
    subset_df = df_text.iloc[0:12].copy()

    # Reemplazar 'S/D' por NaN y convertir los valores
    for year in ['2021', '2022', '2023', '2024', '2025']:
        subset_df[year].replace('S/D', np.nan, inplace=True)
        subset_df[year].replace('N/A', np.nan, inplace=True)  # También manejar 'N/A'
        
        if sistema_viejo:
            # Sistema viejo: eliminar comas como separador de miles
            subset_df[year] = subset_df[year].astype(str).str.replace(',', '', regex=False)
        else:
            # Sistema nuevo: eliminar todos los puntos (separadores de miles) y todo después de la coma decimal
            subset_df[year] = subset_df[year].astype(str).str.replace(r'\.', '', regex=True)  # Eliminar puntos
            subset_df[year] = subset_df[year].str.replace(r',.*', '', regex=True)  # Eliminar coma y todo lo después

        subset_df[year] = pd.to_numeric(subset_df[year], errors='coerce').astype('Int64')

    ventas_records = []
    meses = subset_df['Mes'].tolist()

    # Crear registros solo cuando existe dato (omitiendo NaN)
    for i, mes in enumerate(meses):
        for year in ['2021', '2022', '2023', '2024', '2025']:
            if not pd.isnull(subset_df[year].iloc[i]):
                ventas_records.append([f"{mes.capitalize()} - {year}", subset_df[year].iloc[i]])

    df_ventas = pd.DataFrame(ventas_records, columns=['Mes', 'Ventas'])
    return df_ventas

pd.set_option('display.float_format', lambda x: '%.0f' % x)

def main():
    st.write("<h1 style='text-align: center;'>Deuda📊 vs 📈Ventas</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'><small>Desarrollado por JSantacecilia - JSaborido - Equipo Agro</small></p>",
        unsafe_allow_html=True
    )

    # Interruptor para seleccionar el sistema
    sistema_viejo = st.radio(
        "Selecciona el formato de ventas (Pitagoras - SIPF):",
        ["Ambos sistemas"],
        horizontal=True,
        index=0  # Por defecto sistema viejo
    )
    
    # Convertir a booleano
    sistema_viejo = (sistema_viejo == "Sistema Viejo")

    uploaded_file = st.file_uploader("Sube archivo DEUDA", type=["xls", "xlsx"])
    df_merged = None
    if uploaded_file is not None:
        df_merged = process_file(uploaded_file)
        
    text_input = st.text_area("Pega aquí EVOLUCIÓN DE VENTAS")

    if text_input:
        try:
            df_ventas = process_text_input(text_input, sistema_viejo)
            if df_merged is not None:
                # Extraer solo el nombre del mes (sin año) para la fusión
                df_ventas['Mes'] = df_ventas['Mes'].apply(lambda x: x.split(' - ')[0])
                df_final = df_merged.merge(process_text_input(text_input, sistema_viejo), on='Mes', how='left')
                df_final['dolares'] = (df_final['Ventas'] / df_final['iipc']) / df_final['dolar']
                df_selected_columns = df_final[['Mes', 'ipc', 'dolar', 'Deuda', 'Ventas']]
                
                grafico(df_final)
                graficomm(df_final)
                graficodol(df_final)
                graficobar(df_final)
                
                with st.expander("Tabla de control"):
                    st.dataframe(df_selected_columns, use_container_width=True)
            else:
                st.dataframe(df_ventas)
        except Exception as e:
            st.error(f"Error procesando los datos: {e}")

    with st.expander("Metodología utilizada"):
        st.write("Para la deuda a valores constantes se multiplica deuda histórica de Nosis por Indices IPC Cobertura Nacional suministrado por INDEC, último informe Octubre 2025. Para deuda en dólares se divide deuda histórica de Nosis por dólar mayorista (cotización del último día hábil del mes en cuestión).")
        st.write("Para las ventas se utiliza las generadas por Pitagoras ajustadas. En el caso de la SMA3 se utilizan medias móviles (el valor graficado es el resultado del promedio entre el valor de dicho mes y el de los dos meses anteriores).")
        st.write("Utilizar una media móvil de ventana 3 al graficar ventas es beneficioso porque suaviza las fluctuaciones y reduce el ruido, permitiendo identificar tendencias subyacentes de manera más clara y precisa.")

if __name__ == "__main__":
    main()