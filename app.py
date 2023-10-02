import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# -------------- Settings ------------------------
page_title= 'Detección de transacciones fraudulentas'
page_icon= ':money_with_wings:'
icon_header= ':credit_card:'
icon_name= ':v:'
layout = 'centered'

# ------------ Page Configuration ----------------
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + ' ' + page_icon)

# ----------- Navigation Menu --------------------
selected = option_menu(
    menu_title=None,
    options=['Comprobar transacción', 'Transacciones históricas'],
    icons=['check-circle', 'coin'],
    orientation='horizontal'
)

# ----------- Hide Streamlit Style ---------------
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------------------- PAGE 1 --------------------------------------
      
# ------------------- Load Model --------------------------

st.set_option('deprecation.showfileUploaderEncoding', False)
st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/cnn_model.hdf5')
    return model
model = load_model()

# ------------------- Data Processing --------------------------

scaler = StandardScaler()

historical = pd.read_csv('Final Transactions.csv')

columns_to_scale = ['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS']

scaler.fit(historical[columns_to_scale])

def preprocess_input(Cliente_ID, Terminal_ID, Cantidad, Tiempo):
    Cliente_ID = float(Cliente_ID)
    Terminal_ID = float(Terminal_ID)
    Cantidad = float(Cantidad)
    Tiempo = float(Tiempo)
    
    x_in = np.array([Cliente_ID, Terminal_ID, Cantidad, Tiempo]).reshape(1, -1)
    
    x_in_scaled = scaler.transform(x_in)
    
    return x_in_scaled

# ------------------------- Form ------------------------------
if selected == 'Comprobar transacción':
    st.header(f'Ingresa los datos {icon_header}')
    with st.form('entry_form', clear_on_submit= True):
        Cliente_ID = st.text_input('ID Cliente:', key='Customer ID')
        Terminal_ID = st.text_input('ID Terminal:', key='Terminal ID')
        Cantidad = st.text_input('Cantidad (USD):', key='Amount')
        Tiempo = st.text_input('Tiempo (segundos):', key='Time in seconds')
        submitted = st.form_submit_button('Comprobar')

    if submitted:
       
        x_in = preprocess_input(Cliente_ID, Terminal_ID, Cantidad, Tiempo)
        prediction = model.predict(x_in)
        
        if prediction[0][0] >= 0.5:
            st.success('La transacción es: Fraudulenta'.upper())
        else:
            st.success('La transacción es: Legítima'.upper()) 
            
# --------------------------- PAGE 2 --------------------------------------

# Importar data
historical = pd.read_csv('Final Transactions.csv')

# --------------- Show Transactions -------------------------------
if selected == 'Transacciones históricas':
        
    st.header('Transacciones Históricas')
    
    total_transacciones =  len(historical)
    fraudes = len(historical.loc[historical['TX_FRAUD'] == 1])
    
    col1, col2, col3 = st.columns(3)
    col1.metric('Total de Trasacciones', f'{total_transacciones:,.0f}')
    col2.metric('Fraudes Detectados', f'{fraudes:,.0f}')
    sel_val = col3.multiselect('Tipo de Transacción', sorted(historical['TX_FRAUD'].unique()))

# ----------------------- Show Pie Chart ---------------------------
#Comprobamos balance del dataset
    labels = ['Legítimas', 'Fraudulentas']
    sizes = historical['TX_FRAUD'].value_counts()
        
    pie_data = pd.DataFrame({'labels' : labels, 'sizes' : sizes})
    
    fig = px.pie(pie_data, names=labels, values='sizes', 
                 color_discrete_map={'Legítimas' :'royalblue', 'Fraudulentas' : 'lightcyan'},
                 labels= {'labels' : 'Tipo de Transacción'})
    st.plotly_chart(fig)
            
# ----------------------- Show Dataframe ----------------------------
    def filter_data(df, sel_val):
        df_copy = df.copy()
        
        if len(sel_val) > 0:
            df_copy = df_copy[df_copy['TX_FRAUD'].isin(sel_val)]
            
        return df_copy
    
    df_ = filter_data(historical, sel_val)
    
    st.dataframe(df_, column_config= {
        'CUSTOMER_ID': 'ID Cliente',
        'TERMINAL_ID': 'ID Terminal',
        'TX_AMOUNT': 'Cantidad (USD)',
        'TX_TIME_SECONDS' : 'Tiempo (segundos)',
        'TX_FRAUD': 'Fraudes',
        'Unnamed: 0': None,
        'TRANSACTION_ID': None,
        'TX_DATETIME': None,
        'TX_TIME_DAYS': None,
        'TX_FRAUD_SCENARIO': None     
    }, 
    hide_index=True, use_container_width=True)
    st.column_config.NumberColumn('Cantidad (USD)', format='$ %d')
    
    
        
        