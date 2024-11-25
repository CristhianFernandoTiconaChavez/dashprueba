import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#PERSONALIZAR PÁGINA
st.set_page_config(page_title='Dashboard', page_icon="🎅🏼", layout='wide')

#DATASETS
data_anuncios = pd.read_csv("anuncios.csv")
data_campañas = pd.read_csv("campañas.csv")
data_empresas = pd.read_csv("empresas.csv")
data_productos = pd.read_csv("productos.csv")

# COMBINANDO DATASETS POR CLAVE COMÚN (idAd)
data_combinada = pd.merge(data_anuncios, data_productos, on="idAd")

columna1, columna2 = st.columns([0.4, 0.6])
with columna1:
    st.subheader("📱 MARKETING DIGITAL")
    #ESTADÍSTICAS DESCRIPTIVAS
    plataforma_eficaz = data_anuncios.groupby('plataforma')['clicks'].sum().idxmax()
    tipo_producto_popular = str(data_productos['nomProd'].mode()[0])
    tipo_de_empresa_mas_comun = str(data_empresas['tipo'].mode()[0])
    campaña_con_mayor_inversion = data_campañas.groupby('nomCamp')['presupuesto'].sum().idxmax()

    est1, est2 = st.columns(2, gap='large')
    with est1: 
        st.info('Plataforma', icon='📈')
        st.metric(label='Plataforma más popular para anuncios', value=plataforma_eficaz)
        st.info('Producto', icon='📈')
        st.metric(label='Producto más popular para compras en anuncios', value=tipo_producto_popular)
    with est2:
        st.info('Empresa', icon='📈')
        st.metric(label='Tipo de empresa más común', value=tipo_de_empresa_mas_comun)        
        st.info('Campaña', icon='📈')
        st.metric(label='Campaña con mayor inversión', value=campaña_con_mayor_inversion)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Datos combinados
    #st.subheader("Datos combinados:")
    #st.dataframe(data_combinada)

    #####################################################################
    # PREDICCIÓN BAYESIANA
    #####################################################################

    st.title("Predicción Bayesiana")

    # Preparar los datos para el modelo
    #st.subheader("Entrenamiento del modelo:")

    # Crear las características (X) y la variable objetivo (y)
    X = data_combinada[["clicks", "visualizacion", "precio"]]
    mediana_adquisicion = data_combinada["adquisicion"].median()  # Calcular la mediana
    y = data_combinada["adquisicion"] > mediana_adquisicion  # Clasificar como alta o baja adquisición

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entrenar modelo Naive Bayes
    modelo_bayes = GaussianNB()
    modelo_bayes.fit(X_train, y_train)

    # Predicción
    y_pred = modelo_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Mostrar resultados
    #st.metric("Precisión del modelo", f"{accuracy*100:.2f}%")
    # Predicción interactiva
    st.subheader("Simulador de predicción")
    input_clicks = st.slider("Clics", min_value=0, max_value=500, value=100)
    input_visualizacion = st.slider("Visualizaciones", min_value=0, max_value=3000, value=1000)
    input_precio = st.slider("Precio del Producto", min_value=0, max_value=1500, value=500)

    prediccion = modelo_bayes.predict([[input_clicks, input_visualizacion, input_precio]])
    resultado = "Alta Adquisición" if prediccion[0] else "Baja Adquisición"
    st.write(f"Resultado predicho: **{resultado}**")
    #####################################################################
    # PREDICCIÓN BAYESIANA
    #####################################################################    


with columna2:
    columna3, columna4 = st.columns([0.5, 0.5])
    
    with columna3:
        ####################################################
        #BARRAS
        ####################################################
        #anuncios
        with st.expander("Anuncios"):
            mostrar_anuncios = st.multiselect('Filter: ', data_anuncios.columns, default=[])
            st.write(data_anuncios[mostrar_anuncios])
        st.subheader("Distribución de Clics por Plataforma")
        # Crear un gráfico de dona
        dona_plataforma = px.pie(
            data_anuncios,
            values="clicks",
            names="plataforma",
            hole=0.4,  # Hace el gráfico de dona
            title="Clics por Plataforma"
        )
        st.plotly_chart(dona_plataforma)

        #campañas
        with st.expander("Campañas"):
            mostrar_campañas = st.multiselect('Filter: ', data_campañas.columns, default=[])
            st.write(data_campañas[mostrar_campañas])        
        campañas_figura = px.bar(data_campañas, x='nomCamp', y='presupuesto', color='nomCamp', title='Presupuestos de las Campañas') 
        st.plotly_chart(campañas_figura)


    with columna4:
        ####################################################
        #BARRAS
        ####################################################
        #productos
        with st.expander("Productos"):
            mostrar_productos = st.multiselect('Filter: ', data_productos.columns, default=[])
            st.write(data_productos[mostrar_productos])        
        productos_figura = px.bar(data_productos, x='nomProd', y='precio', color='nomProd', title='Precios por Producto') 
        st.plotly_chart(productos_figura)        
        #empresas
        with st.expander("Empresas"):
            mostrar_empresas = st.multiselect('Filter: ', data_empresas.columns, default=[])
            st.write(data_empresas[mostrar_empresas])        
        empresas_figura = px.bar(data_empresas, x='nomEmp', y='tipo', color='nomEmp', title='Tecnologías de Empresas') 
        st.plotly_chart(empresas_figura) 
        
        #Gráfico Relación entre Precio y Adquisiciones
        st.subheader("Relación entre Precio y Adquisiciones")
        graf_precio_adquisiciones = px.scatter(data_combinada, x="precio", y="adquisicion", color="categoria", title="Precio vs Adquisiciones")
        st.plotly_chart(graf_precio_adquisiciones)              
    












#    ################################################################################################################################################
#
#    # Crear la columna 'Resultado'
#    data_anuncios["Resultado"] = np.where(data_anuncios["Número de Clicks"] > 50, 1, 0)
#    st.write("Dataset con la columna 'Resultado':")
#    st.write(data_anuncios)
#    from sklearn.model_selection import train_test_split
#    from sklearn.naive_bayes import GaussianNB
#    from sklearn.metrics import classification_report
#
#    # Seleccionar las variables explicativas y objetivo
#    X = data_anuncios[["Número de Anuncios", "Número de Skips", "Costo por Anuncio ($)"]]
#    y = data_anuncios["Resultado"]
#
#    # Dividir en conjuntos de entrenamiento y prueba
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#    # Entrenar el modelo Bayesiano
#    model = GaussianNB()
#    model.fit(X_train, y_train)
#
#    # Hacer predicciones
#    y_pred = model.predict(X_test)
#
#    # Mostrar las métricas del modelo
#    st.subheader("Métricas del Modelo Bayesiano")
#    st.text(classification_report(y_test, y_pred))
#
#    st.subheader("Predicción del Éxito de un Anuncio")
#
#    # Entradas del usuario
#    num_anuncios = st.number_input("Número de Anuncios", min_value=1, max_value=100, value=10)
#    num_skips = st.number_input("Número de Skips", min_value=0, max_value=1000, value=100)
#    costo = st.number_input("Costo por Anuncio ($)", min_value=10, max_value=500, value=100)
#
#    # Crear un dataframe con los valores ingresados
#    nuevo_anuncio = pd.DataFrame({
#        "Número de Anuncios": [num_anuncios],
#        "Número de Skips": [num_skips],
#        "Costo por Anuncio ($)": [costo]
#    })
#
#    # Predicción del modelo
#    prediccion = model.predict(nuevo_anuncio)[0]
#    resultado = "Exitoso" if prediccion == 1 else "No Exitoso"
#
#    st.write(f"El anuncio será: **{resultado}**")
#    fig_resultado = px.box(data_anuncios, x="Resultado", y="Número de Clicks", title="Número de Clicks por Resultado")
#    st.plotly_chart(fig_resultado)
#
#    fig_costo = px.histogram(data_anuncios, x="Costo por Anuncio ($)", color="Resultado", barmode="group", title="Costo por Anuncio según Resultado")
#    st.plotly_chart(fig_costo)
#
#    ################################################################################################################################################