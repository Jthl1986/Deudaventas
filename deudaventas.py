import pandas as pd
import streamlit as st
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="borrador", page_icon="📊")

url = "https://raw.githubusercontent.com/Jthl1986/T1/main/iipcJun24.csv"
df = pd.read_csv(url, encoding='ISO-8859-1', sep=',')

# Ocultar fuente GitHub
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

def grafico(df_final):
    # Función para formatear los números grandes
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.0fM' % (x * 1e-6)

    # Aplicar estilo
    sns.set_style("whitegrid")

    # Crear la figura y el eje
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Graficar la deuda como barras
    barplot = sns.barplot(x='Mes', y='Deuda', data=df_final, palette='Greens_d', ax=ax1, edgecolor='black', alpha=0.8)

    # Agregar contornos a las barras
    for bar in barplot.patches:
        bar.set_edgecolor('black')

    # Graficar las ventas como línea en el mismo eje, usando plt.plot para manejar discontinuidades
    ax1.plot(df_final['Mes'], df_final['Ventas'], color='r', marker='o', markersize=10, linewidth=2.5, linestyle='-', drawstyle='default')

    # Etiquetas y leyendas
    ax1.set_ylabel('Deuda y Ventas (en millones)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticklabels(df_final['Mes'], rotation=90)
    ax1.yaxis.set_major_formatter(FuncFormatter(millions))

    # Ajustar los límites del eje y
    ax1.set_ylim(0, max(df_final['Deuda'].max(), df_final['Ventas'].max(skipna=True)) * 1.1)

    plt.title('Deuda y Ventas por Mes a valores constantes', fontsize=16)

    # Corregir la leyenda para mostrar solo una línea y una barra
    handles, labels = ax1.get_legend_handles_labels()
    line = plt.Line2D([0], [0], color='r', lw=2.5, marker='o', markersize=10)
    bar = plt.Rectangle((0,0),1,1,fc="green", edgecolor='black')
    ax1.legend([line, bar], ['Ventas', 'Deuda'], loc='upper left')

    # Mejorar la estética general del gráfico
    sns.despine(left=True, bottom=True)

    return st.pyplot(fig)

def process_file(file):
    df_excel = pd.read_excel(file, header=None, engine='xlrd')
    celdas_totales = ['E4', 'K4', 'Q4', 'W4', 'AC4', 'AI4', 'AO4', 'AU4', 'BA4', 'BG4', 'BM4', 'BS4', 'BY4', 'CE4', 'CK4', 'CQ4', 'CW4', 'DC4', 'DI4', 'DO4', 'DU4', 'EA4', 'EG4']
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

def process_text_input(text_input):
    data = StringIO(text_input)
    df_text = pd.read_csv(data, sep='\t')
    subset_df = df_text.iloc[0:12].copy()
    
    # Reemplazar 'S/D' por NaN y convertir los valores
    for year in ['2020', '2021', '2022', '2023', '2024']:
        subset_df[year].replace('S/D', np.nan, inplace=True)
        subset_df[year] = subset_df[year].str.replace('.', '', regex=False).astype(float).astype('Int64')
    
    # Lista para almacenar los registros
    ventas_records = []

    # Nombres de meses
    meses = subset_df['Mes'].tolist()

    # Extraer las ventas y agregar a la lista
    for i, mes in enumerate(meses):
        for year in ['2020', '2021','2022', '2023', '2024']:
            if not pd.isnull(subset_df[year].iloc[i]):
                ventas_records.append([f"{mes.capitalize()} - {year}", subset_df[year].iloc[i]])

    # Crear DataFrame a partir de la lista de registros
    df_ventas = pd.DataFrame(ventas_records, columns=['Mes', 'Ventas'])

    return df_ventas

pd.set_option('display.float_format', lambda x: '%.0f' % x)

def main():
    st.title("Deuda vs Ventas Valores constantes y en u$s")
    st.markdown("<small>Desarrollado por JSantacecilia - JSaborido - Equipo Agro</small>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Sube archivo DEUDA", type=["xls", "xlsx"])
    
    df_merged = None
    if uploaded_file is not None:
        df_merged = process_file(uploaded_file)
        
    
    text_input = st.text_area("Pega aquí Evolución de Ventas")

    if text_input:
        try:
            df_ventas = process_text_input(text_input)
            if df_merged is not None:
                df_ventas['Mes'] = df_ventas['Mes'].apply(lambda x: x.split(' - ')[0])  # Ajustar si es necesario
                df_final = df_merged.merge(process_text_input(text_input), on='Mes', how='left')
                grafico(df_final)
            else:
                st.dataframe(df_ventas)
        except Exception as e:
            st.error(f"Error procesando los datos: {e}")

    with st.expander("Metodología utilizada"):
        st.write("Para la deuda a valores constantes se multiplica deuda histórica de Nosis por Indices IPC Cobertura Nacional suministrado por INDEC, ultimo informe mayo 2024. Para deuda en dólares se divide deuda histórica de Nosis por dolar mayorista cotización del último día hábil del mes")

if __name__ == "__main__":
    main()