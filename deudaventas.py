def graficobar(df_final):
    def millions(x, pos):
        return '%1.0fM' % (x * 1e-6)
    
    # Convertir las fechas a formato datetime y extraer mes y año
    df_final['fecha'] = pd.to_datetime(df_final['fecha'], format='%m/%Y')
    df_final['Mes'] = df_final['fecha'].dt.strftime('%B')
    df_final['Año'] = df_final['fecha'].dt.year

    df_ventas = df_final[['Mes', 'Año', 'Ventas']]

    # Orden cronológico de los meses en inglés
    meses_ordenados = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    df_ventas['Mes'] = pd.Categorical(df_ventas['Mes'], categories=meses_ordenados, ordered=True)
    
    # Crear pivot y ordenar años
    df_pivot = df_ventas.pivot(index='Mes', columns='Año', values='Ventas')
    df_pivot = df_pivot.sort_index(axis=1)  # Ordenar años ascendentemente

    # Verificar si hay al menos dos años con datos
    if len(df_pivot.columns) < 2:
        return  # No graficar si hay menos de dos años

    # Filtrar meses que tengan datos en todos los años
    df_comparison = df_pivot.dropna()
    
    if df_comparison.empty:
        return  # No hay datos válidos para comparar

    # Graficar todos los años disponibles
    fig, ax = plt.subplots(figsize=(12, 8))
    df_comparison.plot(kind='bar', ax=ax)
    
    ax.set_title(f'Comparación de Ventas por Mes ({", ".join(map(str, df_comparison.columns))})')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Ventas')
    ax.set_xticklabels(df_comparison.index, rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.legend(title='Año', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)