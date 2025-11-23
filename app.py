import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, dcc, html, dash_table, Input, Output

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard del Proyecto Final"
server = app.server  

#============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "modelos_finales_dataviz.joblib")
final_models = joblib.load(file_path)
"""
#============================================================

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
    "income"
]

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "adult.csv")
df = pd.read_csv(file_path)
df.columns = columns

#==============================================================

binarias = ['sex']
numericas = ['capital.gain', 'capital.loss', 'hours.per.week', 'age']
categoricas = ['workclass', 'education', 'marital.status', 'occupation', 'race','relationship']

#==============================================================

cols = binarias + categoricas + numericas

for col in cols:
    moda = df[col].mode()[0]
    df[col] = df[col].replace(" ?", moda)

X = df[binarias+categoricas+numericas]
y = df["income"]

map_bin = {
    ' >50K': 1, 
    ' <=50K': 0, 
}

y = y.replace(map_bin)

#=============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
def get_database_connection():
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(connection_string)

# Función para cargar datos
def load_data():
    engine = get_database_connection()
    query = "SELECT * FROM adult_income_data;"
    return pd.read_sql(query, engine)

#========================================================================================================================

df = load_data()
print(f"Datos cargados: {len(df)} registros")

binarias = ['sex']
numericas = ['capital.gain', 'capital.loss', 'hours.per.week', 'age']
categoricas = ['workclass', 'education', 'marital.status', 'occupation', 'race','relationship']

X = df[binarias+categoricas+numericas]
y = df["income"]

map_bin = {
    ' >50K': 1, 
    ' <=50K': 0, 
}

y = y.replace(map_bin)

#=============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#SUBTABS METODOLOGIA========================================================

subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
    html.Div([
        html.H2('Definición del Problema', className="text-center my-4"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4('Contexto del Proyecto', className="text-center mb-4", style={'color': '#2c3e50'}),
                    html.Hr(style={'borderTop': '3px solid #3498db', 'width': '50%', 'margin': '0 auto 20px auto'}),
                    
                    html.P('Predecir si el ingreso anual de un individuo supera los 50,000 dólares basándose en características demográficas, educativas y laborales del Censo de EE.UU.', 
                          style={'textAlign': 'justify', 'padding': '15px', 'fontSize': '16px', 'lineHeight': '1.6'})
                          
                ], style={
                    'padding': '30px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '10px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'marginBottom': '30px'
                })
            ], width=12),
        ]),
        
        dbc.Row([
            # Tipo de Problema
            dbc.Col([
                html.Div([
                    html.H5('Tipo de Problema', className="text-center", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Hr(style={'borderTop': '2px solid #27ae60', 'width': '30%', 'margin': '0 auto 15px auto'}),
                    html.P('Clasificación Binaria Supervisada', 
                          className="text-center", 
                          style={'fontSize': '16px', 'color': '#34495e', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '8px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6, className="mb-4"),
            
            # Variable Objetivo
            dbc.Col([
                html.Div([
                    html.H5('Variable Objetivo', className="text-center", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Hr(style={'borderTop': '2px solid #e74c3c', 'width': '30%', 'margin': '0 auto 15px auto'}),
                    html.P('income (≤50K vs >50K)', 
                          className="text-center", 
                          style={'fontSize': '16px', 'color': '#34495e', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '8px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6, className="mb-4"),
        ]),
        
        dbc.Row([
            # Alcance
            dbc.Col([
                html.Div([
                    html.H5('Alcance', className="text-center", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Hr(style={'borderTop': '2px solid #f39c12', 'width': '30%', 'margin': '0 auto 15px auto'}),
                    html.P('Análisis predictivo utilizando datos del Censo de EE.UU. (Dataset Adult)', 
                          className="text-center", 
                          style={'fontSize': '16px', 'color': '#34495e', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '8px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6, className="mb-4"),
            
            # Aplicación
            dbc.Col([
                html.Div([
                    html.H5('Aplicación', className="text-center", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Hr(style={'borderTop': '2px solid #9b59b6', 'width': '30%', 'margin': '0 auto 15px auto'}),
                    html.P('Herramientas de apoyo para políticas públicas, análisis de movilidad social y estudios socioeconómicos', 
                          className="text-center", 
                          style={'fontSize': '16px', 'color': '#34495e', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '8px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6, className="mb-4"),
        ])
    ])
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    dcc.Tab(label='b. Preparación de Datos', children=[
        html.Div(className='container-fluid', children=[
            html.H4('Preparación de los Datos', className='text-center mb-4'),
            
            # PREPROCESAMIENTO PIPELINE - ARRIBA DE TODO
            html.Div(className='card mb-4', children=[
                html.Div(className='card-header bg-primary text-white', children=[
                    html.H5('Preprocesamiento Pipeline', className='mb-0 text-center')
                ]),
                html.Div(className='card-body', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6('1. Codificación', style={'color': '#e74c3c', 'textAlign': 'center'}),
                                html.P('One-Hot Encoding', style={'textAlign': 'center', 'marginBottom': '5px'}),
                                html.P('Variables categóricas', style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
                            ], style={
                                'textAlign': 'center', 
                                'padding': '20px', 
                                'backgroundColor': '#f3f3f3', 
                                'borderRadius': '10px',
                                'border': '2px solid #A3A3A3',
                                'height': '100%'
                            })
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H6('2. Escalado', style={'color': '#f39c12', 'textAlign': 'center'}),
                                html.P('StandardScaler', style={'textAlign': 'center', 'marginBottom': '5px'}),
                                html.P('Normalización', style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
                            ], style={
                                'textAlign': 'center', 
                                'padding': '20px', 
                                'backgroundColor': "#f3f3f3", 
                                'borderRadius': '10px',
                                'border': '2px solid #A3A3A3',
                                'height': '100%'
                            })
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H6('3. Modelado', style={'color': '#27ae60', 'textAlign': 'center'}),
                                html.P('Random Forest', style={'textAlign': 'center', 'marginBottom': '5px'}),
                                html.P('XGBoost, Logistic', style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
                            ], style={
                                'textAlign': 'center', 
                                'padding': '20px', 
                                'backgroundColor': '#f3f3f3', 
                                'borderRadius': '10px',
                                'border': '2px solid #A3A3A3',
                                'height': '100%'
                            })
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H6('4. Validación', style={'color': '#3498db', 'textAlign': 'center'}),
                                html.P('Group K-Fold', style={'textAlign': 'center', 'marginBottom': '5px'}),
                                html.P('K=5 folds', style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
                            ], style={
                                'textAlign': 'center', 
                                'padding': '20px', 
                                'backgroundColor': "#f3f3f3", 
                                'borderRadius': '10px',
                                'border': '2px solid #A3A3A3',
                                'height': '100%'
                            })
                        ], width=3),
                    ])
                ])
            ]),
            
            dbc.Row([
                # COLUMNA IZQUIERDA - Limpieza y Características
                dbc.Col([
                    # Limpieza y Transformación de Datos
                    html.Div(className='card mb-4', children=[
                        html.Div(className='card-header bg-success text-white', children=[
                            html.H5('Limpieza y Transformación de Datos', className='mb-0')
                        ]),
                        html.Div(className='card-body', children=[
                            html.Ul([
                                html.Li([
                                    html.Strong('Dataset: '), 
                                    'Adult Census Income (32560 instancias, 14 características)'
                                ]),
                                html.Li([
                                    html.Strong('Manejo de valores missing: '), 
                                    'Imputación de datos faltantes en variables categóricas con la moda'
                                ]),
                                html.Li([
                                    html.Strong('Codificación de variables: '), 
                                    'One-Hot Encoding para variables categóricas'
                                ]),
                                html.Li([
                                    html.Strong('Transformación de variable objetivo: '), 
                                    'Mapeo a valores binarios (≤50K:0, >50K:1)'
                                ]),
                                html.Li([
                                    html.Strong('Variables numéricas: '), 
                                    'Estandarización para modelos sensibles a escalas'
                                ]),
                                html.Li([
                                    html.Strong('División del dataset: '), 
                                    '80% entrenamiento (26048), 20% prueba (6512)'
                                ])
                            ])
                        ])
                    ]),
                    
                    # Características Utilizadas
                    html.Div(className='card', children=[
    html.Div(className='card-header bg-info text-white', children=[
        html.H5('Características Utilizadas', className='mb-0')
    ]),
        html.Div(className='card-body', children=[
            # Demográficas
            dbc.Row([
                dbc.Col([
                    html.Img(src='assets/demograficas.png', 
                            style={'width': '50px', 'height': '50px', 'display': 'block', 'margin': '0 auto'})
                ], width=2),
                dbc.Col([
                    html.Li([html.Strong('Demográficas:'), ' age, sex, race'],
                        style={'listStyle': 'none', 'padding': '8px 0'})
                ], width=10)
            ], className="align-items-center mb-2"),
            
            # Educación
            dbc.Row([
                dbc.Col([
                    html.Img(src='assets/educacion.png', 
                            style={'width': '50px', 'height': '50px', 'display': 'block', 'margin': '0 auto'})
                ], width=2),
                dbc.Col([
                    html.Li([html.Strong('Educación:'), ' education'],
                        style={'listStyle': 'none', 'padding': '8px 0'})
                ], width=10)
            ], className="align-items-center mb-2"),
            
            # Laborales
            dbc.Row([
                dbc.Col([
                    html.Img(src='assets/laborales.png', 
                            style={'width': '50px', 'height': '50px', 'display': 'block', 'margin': '0 auto'})
                ], width=2),
                dbc.Col([
                    html.Li([html.Strong('Laborales:'), ' occupation, workclass, hours.per.week'],
                        style={'listStyle': 'none', 'padding': '8px 0'})
                ], width=10)
            ], className="align-items-center mb-2"),
            
            # Familiares
            dbc.Row([
                dbc.Col([
                    html.Img(src='assets/familiares.png', 
                            style={'width': '50px', 'height': '50px', 'display': 'block', 'margin': '0 auto'})
                ], width=2),
                dbc.Col([
                    html.Li([html.Strong('Familiares:'), ' marital.status, relationship'],
                        style={'listStyle': 'none', 'padding': '8px 0'})
                ], width=10)
            ], className="align-items-center mb-2"),
            
            # Económicas
            dbc.Row([
                dbc.Col([
                    html.Img(src='assets/economicas.png', 
                            style={'width': '50px', 'height': '50px', 'display': 'block', 'margin': '0 auto'})
                ], width=2),
                dbc.Col([
                    html.Li([html.Strong('Económicas:'), ' capital.gain, capital.loss'],
                        style={'listStyle': 'none', 'padding': '8px 0'})
                ], width=10)
            ], className="align-items-center")
        ])
    ])
                    
                ], width=6),
                
                # COLUMNA DERECHA - Estrategia de Validación
                dbc.Col([
                    html.Div(className='card', children=[
                        html.Div(className='card-header bg-warning text-dark', children=[
                            html.H5('Estrategia de Validación - Group K-Fold (K=5)', className='mb-0')
                        ]),
                        html.Div(className='card-body', children=[
                            html.P('Para asegurar la robustez del modelo y evitar data leakage, se implementó Group K-Fold validation:',
                                style={'textAlign': 'justify', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                            
                            html.Div([
                                html.Table([
                                    # Header de la tabla
                                    html.Tr([
                                        html.Th('Iteración', style={'padding': '12px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center'}),
                                        html.Th('Folds de Entrenamiento', style={'padding': '12px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center'}),
                                        html.Th('Fold de Validación', style={'padding': '12px', 'backgroundColor': '#3498db', 'color': 'white', 'textAlign': 'center'})
                                    ]),
                                    # Filas de datos
                                    html.Tr([
                                        html.Td('1', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                        html.Td('2, 3, 4, 5', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#e8f5e8'}),
                                        html.Td('1', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#ffe8e8', 'fontWeight': 'bold'})
                                    ]),
                                    html.Tr([
                                        html.Td('2', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                        html.Td('1, 3, 4, 5', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#e8f5e8'}),
                                        html.Td('2', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#ffe8e8', 'fontWeight': 'bold'})
                                    ]),
                                    html.Tr([
                                        html.Td('3', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                        html.Td('1, 2, 4, 5', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#e8f5e8'}),
                                        html.Td('3', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#ffe8e8', 'fontWeight': 'bold'})
                                    ]),
                                    html.Tr([
                                        html.Td('4', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                        html.Td('1, 2, 3, 5', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#e8f5e8'}),
                                        html.Td('4', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#ffe8e8', 'fontWeight': 'bold'})
                                    ]),
                                    html.Tr([
                                        html.Td('5', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}),
                                        html.Td('1, 2, 3, 4', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#e8f5e8'}),
                                        html.Td('5', style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': '#ffe8e8', 'fontWeight': 'bold'})
                                    ])
                                ], style={
                                    'width': '100%', 
                                    'border': '2px solid #3498db',
                                    'borderCollapse': 'collapse',
                                    'margin': '20px 0',
                                    'fontFamily': 'Arial, sans-serif'
                                })
                            ]),
                            
                            html.H6('Ventajas del Group K-Fold:', style={'color': '#27ae60', 'marginTop': '20px'}),
                            html.Ul([
                                html.Li('Reduce el sobreajuste (overfitting) mediante validación robusta'),
                                html.Li('Utiliza todos los datos tanto para entrenamiento como para validación'),
                                html.Li('Proporciona una estimación más confiable del rendimiento del modelo'),
                                html.Li('Mantiene la integridad de grupos naturales en los datos')
                            ], style={'paddingLeft': '20px'})
                            
                        ], style={
                            'padding': '25px',
                            'borderRadius': '10px',
                            'backgroundColor': '#ffffff'
                        })
                    ])
                    
                ], width=6)
            ])
        ])
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('Modelos seleccionados', style={'textAlign': 'center', 'marginBottom': '40px'}),
        
        dbc.Row([
            # Regresión Logística
            dbc.Col([
                html.Div([
                    html.H4('Regresión Logística', style={'color': '#e74c3c', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Modelo lineal para clasificación probabilística', 
                          style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '15px', 'minHeight': '50px'}),
                    html.P('Interpretabilidad y línea base para comparación de modelos complejos',
                          style={'textAlign': 'center', 'fontStyle': 'italic', 'backgroundColor': '#fdf2f2', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px', 'minHeight': '80px'}),
                    html.Div([
                        html.Img(src='assets/regresion.png', 
                                style={'width': '100%', 'maxWidth': '400px', 'border': '3px solid #A3A3A3', 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto'})
                    ], className="text-center")
                ], style={
                    'padding': '25px', 
                    'backgroundColor': '#ffffff', 
                    'borderRadius': '15px',
                    'border': '2px solid #A3A3A3',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=4),

            # Random Forest
            dbc.Col([
                html.Div([
                    html.H4('Random Forest', style={'color': '#27ae60', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Ensemble basado en árboles de decisión con bagging', 
                          style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '15px', 'minHeight': '50px'}),
                    html.P('Robustez ante overfitting y capacidad de manejar relaciones no lineales',
                          style={'textAlign': 'center', 'fontStyle': 'italic', 'backgroundColor': '#f0f8f4', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px', 'minHeight': '80px'}),
                    html.Div([
                        html.Img(src='assets/random.png', 
                                style={'width': '100%', 'maxWidth': '400px', 'border': '3px solid #A3A3A3', 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto'})
                    ], className="text-center")
                ], style={
                    'padding': '25px', 
                    'backgroundColor': '#ffffff', 
                    'borderRadius': '15px',
                    'border': '2px solid #A3A3A3',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=4),
            
            # XGBoost
            dbc.Col([
                html.Div([
                    html.H4('XGBoost', style={'color': '#3498db', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.P('Algoritmo de boosting con optimización de gradiente', 
                          style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '15px', 'minHeight': '50px'}),
                    html.P('Alto rendimiento en competencias y manejo eficiente de variables categóricas',
                          style={'textAlign': 'center', 'fontStyle': 'italic', 'backgroundColor': '#f0f8ff', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px', 'minHeight': '80px'}),
                    html.Div([
                        html.Img(src='assets/xgboost.png', 
                                style={'width': '100%', 'maxWidth': '400px', 'border': '3px solid #A3A3A3', 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto'})
                    ], className="text-center")
                ], style={
                    'padding': '25px', 
                    'backgroundColor': '#ffffff', 
                    'borderRadius': '15px',
                    'border': '2px solid #A3A3A3',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=4)
        ], className="mb-4")
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('Entrenamiento y Evaluación del Modelo', style={'textAlign': 'center', 'marginBottom': '40px'}),
        
        dbc.Row([
            # Columna izquierda - Proceso de Entrenamiento
            dbc.Col([
                html.Div([
                    html.H5('Proceso de Entrenamiento', style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '25px'}),
                    html.Hr(style={'borderTop': '3px solid #3498db', 'width': '50%', 'margin': '0 auto 20px auto'}),
                    
                    html.Div([
                        html.H6('Pipeline Unificado', style={'color': '#3498db', 'marginBottom': '10px'}),
                        html.P('Preprocesamiento + Modelo en un flujo integrado', 
                              style={'textAlign': 'center', 'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Div([
                        html.H6('Optimización', style={'color': '#e74c3c', 'marginBottom': '10px'}),
                        html.P('GridSearchCV para hiperparámetros', 
                              style={'textAlign': 'center', 'backgroundColor': '#fdf2f2', 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Div([
                        html.H6('Validación', style={'color': '#27ae60', 'marginBottom': '10px'}),
                        html.P('5-fold cross-validation para robustez', 
                              style={'textAlign': 'center', 'backgroundColor': '#f0f8f4', 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Div([
                        html.H6('Balanceo', style={'color': '#9b59b6', 'marginBottom': '10px'}),
                        html.P('Manejo de clases desbalanceadas con class_weight y scale_pos_weight', 
                              style={'textAlign': 'center', 'backgroundColor': '#f8f0f8', 'padding': '10px', 'borderRadius': '8px'})
                    ])
                    
                ], style={
                    'padding': '30px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '15px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6),
            
            # Columna derecha - Validación Utilizada
            dbc.Col([
                html.Div([
                    html.H5('Estrategia de Validación', style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '25px'}),
                    html.Hr(style={'borderTop': '3px solid #e74c3c', 'width': '50%', 'margin': '0 auto 20px auto'}),
                    
                    html.Div([
                        html.H6('Separación Train - Test', style={'color': '#e74c3c', 'marginBottom': '10px'}),
                        html.P('80-20 para evaluación final del modelo', 
                              style={'textAlign': 'center', 'backgroundColor': '#fdf2f2', 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Div([
                        html.H6('Análisis de Errores', style={'color': '#9b59b6', 'marginBottom': '10px'}),
                        html.P('Matrices de confusión y curvas ROC para diagnóstico detallado', 
                            style={'textAlign': 'center', 'backgroundColor': '#f8f0f8', 'padding': '10px', 'borderRadius': '8px'})
                    ]),
                    
                    html.Div([
                        html.H6('Prevención Data Leakage', style={'color': '#3498db', 'marginBottom': '10px'}),
                        html.P('Preprocesamiento dentro de CV folds y transformaciones separadas', 
                            style={'textAlign': 'center', 'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '20px'})
                    ]),
                    
                    html.Div([
                        html.H6('Repetibilidad', style={'color': '#9b59b6', 'marginBottom': '10px'}),
                        html.P('Random state fijo para resultados consistentes', 
                              style={'textAlign': 'center', 'backgroundColor': '#f8f0f8', 'padding': '10px', 'borderRadius': '8px'})
                    ])
                    
                ], style={
                    'padding': '30px',
                    'border': '2px solid #A3A3A3',
                    'borderRadius': '15px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6)
        ], className="mb-4"),
        
        html.H5('Métricas de Evaluación', style={'marginTop': '40px', 'textAlign': 'center'}),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('Accuracy', style={'color': '#27ae60', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.Img(src='assets/accuracy.png', 
                                style={'width': '95%', 'maxWidth': '350px', 'border': '2px solid #A3A3A3', 'borderRadius': '8px', 'display': 'block', 'margin': '5px auto'}),
                        html.P('Proporción total de predicciones correctas',
                            style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '8px', 'marginBottom': '0'})
                    ], style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'height': '100%'})
                ], width=4),
                
                dbc.Col([
                    html.Div([
                        html.H6('Precision', style={'color': '#3498db', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.Img(src='assets/precision.png', 
                                style={'width': '95%', 'maxWidth': '350px', 'border': '2px solid #A3A3A3', 'borderRadius': '8px', 'display': 'block', 'margin': '5px auto'}),
                        html.P('Capacidad de no predecir falsos positivos',
                            style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '8px', 'marginBottom': '0'})
                    ], style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'height': '100%'})
                ], width=4),
                
                dbc.Col([
                    html.Div([
                        html.H6('Recall', style={'color': '#e74c3c', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.Img(src='assets/recall.png', 
                                style={'width': '95%', 'maxWidth': '350px', 'border': '2px solid #A3A3A3', 'borderRadius': '8px', 'display': 'block', 'margin': '5px auto'}),
                        html.P('Capacidad de encontrar todos los positivos reales',
                            style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '8px', 'marginBottom': '0'})
                    ], style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'height': '100%'})
                ], width=4)
            ], className="mb-3"),  # Reducido de mb-4 a mb-3

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('F1-Score', style={'color': '#9b59b6', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.Img(src='assets/f1_score.png', 
                                style={'width': '95%', 'maxWidth': '350px', 'border': '2px solid #A3A3A3', 'borderRadius': '8px', 'display': 'block', 'margin': '5px auto'}),
                        html.P('Media armónica entre precision y recall',
                            style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '8px', 'marginBottom': '0'})
                    ], style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'height': '100%'})
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H6('AUC-ROC', style={'color': '#f39c12', 'textAlign': 'center', 'marginBottom': '10px'}),
                        html.Img(src='assets/auc_roc.png', 
                                style={'width': '95%', 'maxWidth': '350px', 'border': '2px solid #A3A3A3', 'borderRadius': '8px', 'display': 'block', 'margin': '5px auto'}),
                        html.P('Área bajo la curva ROC',
                            style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '8px', 'marginBottom': '0'})
                    ], style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'height': '100%'})
                ], width=6)
            ], className="mb-3")  # Reducido de mb-4 a mb-3
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    })
])

#SUBTABS RESULTADOS========================================================

subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. Análisis Univariado', children=[
        html.H4('Análisis univariado', style={'textAlign': 'center', 'marginBottom': '40px'}),
        html.P('Explora la distribución de variables categóricas y numéricas individualmente.'),
        dbc.Row([
        dbc.Col([
            html.H5("Variables Categóricas"),
            dcc.Dropdown(
                id='eda2-cat-dropdown',
                options=[{'label': v, 'value': v} for v in categoricas+binarias if v not in ["native.country"]],
                value=[v for v in categoricas if v not in ["native.country"]][0],
                clearable=False
            ),
            html.Div(id='eda2-cat-output')
        ], width=6),

        dbc.Col([
            html.H5("Variables Numéricas"),
            dcc.Dropdown(
                id='eda2-num-dropdown',
                options=[{'label': v, 'value': v} for v in numericas if v not in ["fnlwgt"]],
                value=[v for v in numericas if v not in ["fnlwgt"]][0],
                clearable=False
            ),
            html.Div(id='eda2-num-output')
        ], width=6)
    ])
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    dcc.Tab(label='b. Análisis Bivariado', children=[
        html.H4('Análisis Bivariado - Relaciones con variable objetivo', style={'textAlign': 'center', 'marginBottom': '40px'}),
        html.P('Explora las distribuciones de variables numéricas y categóricas.'),
        html.H5("Variables Numéricas"),
        dcc.Dropdown(
            id='dropdown-numerica',
            options=[{'label': c, 'value': c} for c in numericas],
            value=numericas[0],
            clearable=False,
            style={'width': '50%'}
        ),
        html.Div(id='grafico-numerica'),
        html.Hr(),
        html.H5("Variables Categóricas"),
        dcc.Dropdown(
            id='dropdown-categorica',
            options=[{'label': c, 'value': c} for c in categoricas+binarias],
            value=categoricas[0],
            clearable=False,
            style={'width': '50%'}
        ),
        html.Div(id='grafico-categorica')
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('Visualización de Resultados del Modelo', style={'textAlign': 'center', 'marginBottom': '40px'}),
        dcc.Dropdown(
            id='modelo-dropdown',
            options=[{'label': k, 'value': k} for k in final_models.keys()],
            value=list(final_models.keys())[0],
            clearable=False
        ),
        html.Div(id='visualizacion-modelo')
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),

    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('Indicadores de Evaluación del Modelo', style={'textAlign': 'center', 'marginBottom': '40px'}),
        html.Div(id='indicadores-modelo')
    ], style={
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'border': '1px solid #dee2e6',
        'padding': '10px',
        'fontWeight': 'bold'
    },
    selected_style={
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': '1px solid #007bff',
        'padding': '10px',
        'fontWeight': 'bold'
    }),
    
    dcc.Tab(label='e. Limitaciones', children=[
    html.H4('Limitaciones y Consideraciones Finales', style={'textAlign': 'center', 'marginBottom': '40px'}),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5('Limitaciones Identificadas', style={'color': '#e74c3c', 'marginBottom': '25px', 'textAlign': 'center'}),
                html.Hr(style={'borderTop': '3px solid #e74c3c', 'width': '50%', 'margin': '0 auto 20px auto'}),
                
                html.Ul([
                    html.Li([
                        html.Strong('Dataset Histórico: '),
                        'Los datos del Census Income (1994) no reflejan cambios económicos, sociales y tecnológicos de las últimas décadas.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#fdf2f2', 'borderRadius': '5px'}),
                    
                    html.Li([
                        html.Strong('Contexto Temporal: '),
                        'Los niveles de ingreso de 50K dólares en 1994 equivalen aproximadamente a 100K dólares en 2025 debido a la inflación.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#fdf2f2', 'borderRadius': '5px'}),
                    
                    html.Li([
                        html.Strong('Variables Limitadas: '),
                        'Falta información sobre factores contemporáneos como educación digital, teletrabajo y habilidades tecnológicas.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#fdf2f2', 'borderRadius': '5px'}),
                    
                    html.Li([
                        html.Strong('Sesgo Geográfico: '),
                        'Los datos se limitan principalmente a Estados Unidos, limitando la aplicabilidad a otros contextos socioeconómicos.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#fdf2f2', 'borderRadius': '5px'}),
                    
                ], style={'paddingLeft': '0px'})
                
            ], style={
                'padding': '30px',
                'border': '2px solid #e74c3c',
                'borderRadius': '10px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                'height': '100%'
            })
        ], width=6),
        
        dbc.Col([
            html.Div([
                html.H5('Mejoras Futuras', style={'color': '#27ae60', 'marginBottom': '25px', 'textAlign': 'center'}),
                html.Hr(style={'borderTop': '3px solid #27ae60', 'width': '50%', 'margin': '0 auto 20px auto'}),
                
                html.Ul([
                    html.Li([
                        html.Strong('Actualización de Datos: '),
                        'Incluir datos contemporáneos que reflejen la economía digital y nuevas formas de empleo.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#f0f8f4', 'borderRadius': '5px'}),
                    
                    html.Li([
                        html.Strong('Variables Adicionales: '),
                        'Incorporar métricas de habilidades digitales, educación en línea, emprendimiento y trabajo remoto.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#f0f8f4', 'borderRadius': '5px'}),
                    
                    html.Li([
                        html.Strong('Contexto Global: '),
                        'Expandir el análisis a diferentes países y contextos económicos para mayor generalización.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#f0f8f4', 'borderRadius': '5px'}),
                    
                    html.Li([
                        html.Strong('Ajuste por Inflación: '),
                        'Actualizar los umbrales de ingreso según el poder adquisitivo actual para mayor relevancia.'
                    ], style={'marginBottom': '15px', 'textAlign': 'justify', 'padding': '10px', 'backgroundColor': '#f0f8f4', 'borderRadius': '5px'}),
                    
                ], style={'paddingLeft': '0px'})
                
            ], style={
                'padding': '30px',
                'border': '2px solid #27ae60',
                'borderRadius': '10px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                'height': '100%'
            })
        ], width=6)
    ], className="mb-4"),
    
    # Consideraciones finales
    html.Div([
        html.H5('Consideraciones Finales', style={'color': '#3498db', 'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.P('A pesar de las limitaciones temporales del dataset, este proyecto demuestra la viabilidad de aplicar técnicas de machine learning para la predicción de niveles de ingreso. Los modelos desarrollados sirven como base metodológica que puede ser adaptada y mejorada con datos más recientes y variables contemporáneas.',
                  style={'textAlign': 'justify', 'padding': '20px', 'fontSize': '16px', 'lineHeight': '1.6'})
        ], style={
            'backgroundColor': '#f0f8ff',
            'borderRadius': '10px',
            'border': '2px solid #3498db',
            'padding': '10px'
        })
    ], style={'marginTop': '30px'})
], style={
    'backgroundColor': '#f8f9fa',
    'color': '#2c3e50',
    'border': '1px solid #dee2e6',
    'padding': '10px',
    'fontWeight': 'bold'
},
selected_style={
    'backgroundColor': '#007bff',
    'color': 'white',
    'border': '1px solid #007bff',
    'padding': '10px',
    'fontWeight': 'bold'
})
])

#TABS GENERALES========================================================================

tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2('Introducción'),
        html.P('Aquí se presenta una visión general del contexto de la problemática, el análisis realizado y los hallazgos encontrados.'),
        html.P('De manera resumida, indicar lo que se pretende lograr con el proyecto')
    ]),

    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contexto'),
        html.P('Descripción breve del contexto del proyecto.'),
        html.Ul([
            html.Li('Fuente de los datos: Nombre de la fuente'),
            html.Li('Variables de interés: listar variables-operacionalización')
        ])
    ]),

    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        html.P('Describe en pocas líneas la problemática abordada.'),
        html.P('Pregunta problema: ¿Cuál es la pregunta que intenta responder el análisis?')
    ]),

    dcc.Tab(label='4. Objetivos y Justificación', children=[
    html.H2('Objetivos y Justificación', className="text-center my-4"),
    
    dbc.Row([
    dbc.Col([
        html.Div([
            html.H4('Objetivos', className="text-center mb-4", style={'color': '#2c3e50'}),
            html.Hr(style={'borderTop': '3px solid #3498db', 'width': '50%', 'margin': '0 auto 20px auto'}),
            
            html.H5('Objetivo General', className="mt-3", style={'color': '#2c3e50'}),
            html.Div([
                html.P('Desarrollar un sistema de predicción de ingresos que identifique si un individuo supera los 50K dólares anuales mediante la implementación y comparación de múltiples modelos de machine learning optimizados', 
                      style={'textAlign': 'justify', 'padding': '10px'})
            ], style={'backgroundColor': '#f8f9fa', 'borderLeft': '4px solid #3498db', 'padding': '10px', 'marginBottom': '20px'}),
            
            html.H5('Objetivos Específicos', className="mt-4", style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '25px'}),

            # Objetivo 1 - EDA
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='assets/analisis.png', 
                                style={'width': '60px', 'height': '60px', 'display': 'block', 'margin': '0 auto'})
                    ], style={'textAlign': 'center'})
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H6('Análisis Exploratorio', style={'color': '#27ae60', 'marginBottom': '8px'}),
                        html.P('Realizar un EDA exhaustivo del dataset Census Income para identificar patrones demográficos y socioeconómicos asociados a altos ingresos',
                              style={'textAlign': 'justify', 'marginBottom': '0', 'fontSize': '14px'})
                    ])
                ], width=10)
            ], className="align-items-center mb-3", style={'padding': '15px', 'backgroundColor': '#f0f8f4', 'borderRadius': '8px', 'borderLeft': '4px solid #27ae60'}),

            # Objetivo 2 - Implementación
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='assets/implementacion.png', 
                                style={'width': '60px', 'height': '60px', 'display': 'block', 'margin': '0 auto'})
                    ], style={'textAlign': 'center'})
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H6('Implementación de Modelos', style={'color': '#3498db', 'marginBottom': '8px'}),
                        html.P('Implementar y optimizar múltiples algoritmos de clasificación (Random Forest, XGBoost, Logistic Regression) utilizando técnicas de preprocesamiento y validación cruzada',
                              style={'textAlign': 'justify', 'marginBottom': '0', 'fontSize': '14px'})
                    ])
                ], width=10)
            ], className="align-items-center mb-3", style={'padding': '15px', 'backgroundColor': '#f0f8ff', 'borderRadius': '8px', 'borderLeft': '4px solid #3498db'}),

            # Objetivo 3 - Evaluación
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='assets/evaluacion.png', 
                                style={'width': '60px', 'height': '60px', 'display': 'block', 'margin': '0 auto'})
                    ], style={'textAlign': 'center'})
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H6('Evaluación Comparativa', style={'color': '#e74c3c', 'marginBottom': '8px'}),
                        html.P('Evaluar el rendimiento de los modelos mediante métricas robustas (Accuracy, Precision, Recall, F1-Score, AUC-ROC) y seleccionar el mejor enfoque predictivo',
                              style={'textAlign': 'justify', 'marginBottom': '0', 'fontSize': '14px'})
                    ])
                ], width=10)
            ], className="align-items-center", style={'padding': '15px', 'backgroundColor': '#fdf2f2', 'borderRadius': '8px', 'borderLeft': '4px solid #e74c3c'})
            
        ], style={
            'padding': '30px',
            'border': '2px solid #A3A3A3',
            'borderRadius': '10px',
            'backgroundColor': '#ffffff',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'height': '100%'
        })
    ], width=6),
        
        dbc.Col([
            html.Div([
                html.H4('Justificación', className="text-center mb-4", style={'color': '#2c3e50'}),
                html.Hr(style={'borderTop': '3px solid #e74c3c', 'width': '50%', 'margin': '0 auto 20px auto'}),
                
                html.Div([
                    html.P('La predicción de niveles de ingreso tiene aplicaciones significativas en políticas públicas, análisis de movilidad social y estrategias de inclusión financiera. Identificar los factores asociados a mayores ingresos permite diseñar intervenciones más efectivas y comprender mejor las dinámicas socioeconómicas.',
                          style={'textAlign': 'justify', 'padding': '10px'})
                ], style={'backgroundColor': '#f8f9fa', 'borderLeft': '4px solid #e74c3c', 'padding': '10px', 'marginBottom': '25px'}),
                
                # Impacto Científico
                dbc.Row([
                    dbc.Col([
                        html.Img(src='assets/grafico.png', 
                                style={'width': '80px', 'height': '80px', 'display': 'block', 'margin': '0 auto'})
                    ], width=3),
                    dbc.Col([
                        html.H5('Impacto Científico', style={'color': '#3498db', 'marginBottom': '10px'}),
                        html.P('Esta investigación contribuye al campo científico explorando enfoques comparativos entre diferentes familias de algoritmos de clasificación. Mejora la comprensión de las dinámicas no lineales en datos socioeconómicos.',
                              style={'textAlign': 'justify', 'fontSize': '14px', 'marginBottom': '0'})
                    ], width=9)
                ], style={'padding': '15px', 'backgroundColor': "#e4f5ff", 'borderRadius': '8px', 'marginBottom': '15px', 'alignItems': 'center'}),
                
                # Impacto Social
                dbc.Row([
                    dbc.Col([
                        html.Img(src='assets/edificios.png', 
                                style={'width': '80px', 'height': '80px', 'display': 'block', 'margin': '0 auto'})
                    ], width=3),
                    dbc.Col([
                        html.H5('Impacto Social', style={'color': '#C863FF', 'marginBottom': '10px'}),
                        html.P('Al identificar los factores clave asociados con mayores ingresos, el proyecto contribuye a reducir la desigualdad económica. Facilita el diseño de políticas públicas más efectivas y programas de capacitación.',
                              style={'textAlign': 'justify', 'fontSize': '14px', 'marginBottom': '0'})
                    ], width=9)
                ], style={'padding': '15px', 'backgroundColor': "#fceaf7", 'borderRadius': '8px', 'marginBottom': '15px', 'alignItems': 'center'}),
                
                # Impacto Económico
                dbc.Row([
                    dbc.Col([
                        html.Img(src='assets/monedas.png', 
                                style={'width': '80px', 'height': '80px', 'display': 'block', 'margin': '0 auto'})
                    ], width=3),
                    dbc.Col([
                        html.H5('Impacto Económico', style={'color': '#27ae60', 'marginBottom': '10px'}),
                        html.P('La capacidad de predecir potenciales niveles de ingreso permite optimizar recursos en programas de desarrollo, mejorar la focalización de ayudas sociales y diseñar estrategias más eficientes.',
                              style={'textAlign': 'justify', 'fontSize': '14px', 'marginBottom': '0'})
                    ], width=9)
                ], style={'padding': '15px', 'backgroundColor': "#eeffef", 'borderRadius': '8px', 'alignItems': 'center'})
                
            ], style={
                'padding': '30px',
                'border': '2px solid #A3A3A3',
                'borderRadius': '10px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                'height': '100%'
            })
        ], width=6)
    ], className="mb-4")
]),

        dcc.Tab(label='5. Marco Teórico', children=[
            html.H2('Marco Teórico', className="text-center my-4"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4('Conceptos Fundamentales', className="text-center mb-4", style={'color': '#2c3e50'}),
                        html.Hr(style={'borderTop': '3px solid #27ae60', 'width': '50%', 'margin': '0 auto 20px auto'}),
                        
                        html.H5('Machine Learning para Clasificación', style={'color': '#2c3e50', 'marginTop': '20px'}),
                        html.Div([
                            html.P('El machine learning es una rama de la inteligencia artificial que permite a los sistemas aprender patrones de datos. En problemas de clasificación como la predicción de salarios, buscamos asignar etiquetas discretas (≤50K vs >50K).',
                                style={'textAlign': 'justify', 'padding': '15px'})
                        ], style={'backgroundColor': '#f0f8f4', 'borderLeft': '4px solid #27ae60', 'borderRadius': '5px', 'marginBottom': '25px'}),
                        
                        html.H5('Algoritmos de Clasificación', style={'color': '#2c3e50', 'marginTop': '20px'}),
                        html.Ul([
                            html.Li([
                                html.Strong('Random Forest: '),
                                'Método de ensemble que combina múltiples árboles de decisión, reduciendo el sobreajuste mediante promediado'
                            ], style={'marginBottom': '12px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                            html.Li([
                                html.Strong('XGBoost: '),
                                'Algoritmo de boosting que construye árboles secuencialmente, corrigiendo errores de modelos anteriores mediante optimización de gradiente'
                            ], style={'marginBottom': '12px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                            html.Li([
                                html.Strong('Regresión Logística: '),
                                'Modelo lineal probabilístico para clasificación binaria que estima la probabilidad mediante la función sigmoide'
                            ], style={'marginBottom': '12px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                        ], style={'paddingLeft': '10px'}),
                        
                    ], style={
                        'padding': '30px',
                        'border': '2px solid #A3A3A3',
                        'borderRadius': '10px',
                        'backgroundColor': '#ffffff',
                        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                        'height': '100%'
                    })
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H4('Evaluación y Procesamiento', className="text-center mb-4", style={'color': '#2c3e50'}),
                        html.Hr(style={'borderTop': '3px solid #8e44ad', 'width': '50%', 'margin': '0 auto 20px auto'}),
                        
                        html.H5('Métricas de Evaluación', style={'color': '#2c3e50', 'marginTop': '20px'}),
                        html.Ul([
                            html.Li([
                                html.Strong('Accuracy: '),
                                'Proporción de predicciones correctas sobre el total de casos evaluados'
                            ], style={'marginBottom': '10px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f4f6f9', 'borderRadius': '5px'}),
                            html.Li([
                                html.Strong('Precision: '),
                                'Capacidad del modelo de no clasificar como positivo un caso negativo (Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos))'
                            ], style={'marginBottom': '10px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f4f6f9', 'borderRadius': '5px'}),
                            html.Li([
                                html.Strong('Recall: '),
                                'Capacidad del modelo de encontrar todos los casos positivos (Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos))'
                            ], style={'marginBottom': '10px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f4f6f9', 'borderRadius': '5px'}),
                            html.Li([
                                html.Strong('F1-Score: '),
                                'Media armónica entre precision y recall: 2 × (Precision × Recall) / (Precision + Recall)'
                            ], style={'marginBottom': '10px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f4f6f9', 'borderRadius': '5px'}),
                            html.Li([
                                html.Strong('AUC-ROC: '),
                                'Área bajo la curva que mide la capacidad de discriminación del modelo'
                            ], style={'marginBottom': '10px', 'textAlign': 'justify', 'padding': '8px', 'backgroundColor': '#f4f6f9', 'borderRadius': '5px'})
                        ], style={'paddingLeft': '10px'}),
                        
                    ], style={
                        'padding': '30px',
                        'border': '2px solid #A3A3A3',
                        'borderRadius': '10px',
                        'backgroundColor': '#ffffff',
                        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                        'height': '100%'
                    })
                ], width=6)
            ], className="mb-4"),
            
            # Referencias Bibliográficas fuera de las cajas
            html.Div([
                html.H4('Referencias Bibliográficas', className="text-center mb-4", style={'color': '#2c3e50'}),
                html.Hr(style={'borderTop': '3px solid #34495e', 'width': '30%', 'margin': '0 auto 25px auto'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.P('Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.',
                                style={'fontStyle': 'italic', 'marginBottom': '12px', 'textAlign': 'center'}),
                            html.P('Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.',
                                style={'fontStyle': 'italic', 'marginBottom': '12px', 'textAlign': 'center'}),
                            html.P('Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference.',
                                style={'fontStyle': 'italic', 'marginBottom': '12px', 'textAlign': 'center'}),
                            html.P('Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.',
                                style={'fontStyle': 'italic', 'textAlign': 'center'})
                        ], style={
                            'backgroundColor': '#f8f9fa', 
                            'padding': '25px', 
                            'borderRadius': '10px',
                            'border': '2px solid #34495e'
                        })
                    ], width=10, className="mx-auto")
                ])
            ], style={'marginTop': '40px'})
        ]),

    dcc.Tab(label='6. Metodología', children=[
        subtabs_metodologia
    ]),

    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        subtabs_resultados
    ]),

    dcc.Tab(label='8. Conclusiones', children=[
    html.Div([
        html.H2('Conclusiones', className="text-center my-4"),
        
        # Fila 1: Dos columnas (Hallazgos + Selección)
        dbc.Row([
            # Esquina superior izquierda - Hallazgos
            dbc.Col([
                html.Div([
                    html.H4('Principales Hallazgos', className="text-center mb-4", style={'color': '#2c3e50'}),
                    html.Hr(style={'borderTop': '3px solid #3498db', 'width': '50%', 'margin': '0 auto 20px auto'}),
                    html.Ul([
                        html.Li('Random Forest demostró el mejor rendimiento con AUC de 0.940, superando a todos los modelos'),
                        html.Li('XGBoost mostró excelente balance con AUC de 0.937 y buen recall en la clase >50K'),
                        html.Li('Regresión Logística presentó desempeño sólido con AUC de 0.907, considerando su simplicidad'),
                        html.Li('Los tres modelos superaron el 80% de accuracy, validando la viabilidad del enfoque'),
                        html.Li('Random Forest identificó 1439 verdaderos positivos en >50K con alta precisión general')
                    ], style={'textAlign': 'justify', 'lineHeight': '1.6'})
                ], style={
                    'padding': '25px',
                    'border': '2px solid #a3a3a3',
                    'borderRadius': '10px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6),
            
            # Esquina superior derecha - Selección del Modelo
            dbc.Col([
                html.Div([
                    html.H4('Selección del Modelo Óptimo', className="text-center mb-4", style={'color': '#2c3e50'}),
                    html.Hr(style={'borderTop': '3px solid #27ae60', 'width': '50%', 'margin': '0 auto 20px auto'}),
                    html.Ul([
                        html.Li('Random Forest es seleccionado como modelo final debido a su AUC superior (0.940)'),
                        html.Li('La explicabilidad es una característica favor de la elección del modelo'),
                        html.Li('Para aplicaciones críticas, Random Forest ofrece la mejor capacidad discriminativa'),
                        html.Li('XGBoost se considera alternativa robusta con mejor interpretabilidad'),
                        html.Li('La diferencia de 0.03 en AUC entre XGBoost y Random Forest no es tan significativa')
                    ], style={'textAlign': 'justify', 'lineHeight': '1.6'})
                ], style={
                    'padding': '25px',
                    'border': '2px solid #a3a3a3',
                    'borderRadius': '10px',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                })
            ], width=6)
        ], className="mb-4"),
    
    ])
])
]

app.layout = dbc.Container([
    html.H1("Análisis Comparativo de Random Forest, XGBoost y Regresión Logística en Predicción de Ingresos", className="text-center my-4"),
    dcc.Tabs(tabs)
], fluid=True)

#============================== CALLBACK EDA 1
@app.callback(
    Output('grafico-numerica', 'children'),
    Input('dropdown-numerica', 'value')
)
def actualizar_grafico_numerico(variable):
    fig = px.histogram(
        df, x=variable, nbins=30, color='income',
        title=f"Distribución de {variable} por nivel de income",
        barmode='overlay', opacity=0.7
    )
    fig.update_layout(xaxis_title=variable, yaxis_title="Frecuencia")

    desc = df[variable].describe().to_frame().reset_index()
    desc.columns = ['Estadístico', 'Valor']
    table = dash_table.DataTable(
        data=desc.to_dict('records'),
        columns=[{"name": i, "id": i} for i in desc.columns],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )

    return html.Div([
        dcc.Graph(figure=fig),
        html.H6("Resumen estadístico"),
        table
    ])

@app.callback(
    Output('grafico-categorica', 'children'),
    Input('dropdown-categorica', 'value')
)
def actualizar_grafico_categorica(variable):
    cross = df.groupby([variable, 'income']).size().reset_index(name='Frecuencia')

    # Gráfico de barras agrupadas
    fig = px.bar(
        cross,
        x=variable,
        y='Frecuencia',
        color='income',
        barmode='group',
        title=f"Distribución de income por {variable}",
        color_discrete_map={' <=50K': '#1f77b4', ' >50K': '#ff7f0e'}
    )

    fig.update_layout(
        xaxis_title=variable,
        yaxis_title="Frecuencia",
        legend_title="Income",
        height=500,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # --- Tabla de resumen ---
    tabla = pd.crosstab(
        df[variable], df['income']
    ).reset_index().rename(columns={
        ' <=50K': '<=50K (n)',
        ' >50K': '>50K (n)'
    })

    cross_table = dash_table.DataTable(
        data=tabla.to_dict('records'),
        columns=[{"name": i, "id": i} for i in tabla.columns],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        page_size=8
    )

    return html.Div([
        dcc.Graph(figure=fig),
        html.H6("Distribución de income por categoría"),
        cross_table
    ])

#========================================== CALLBACK EDA 2

@app.callback(
    Output('eda2-cat-output', 'children'),
    Input('eda2-cat-dropdown', 'value')
)
def eda2_categorica(var_cat):
    conteos = df[var_cat].value_counts().reset_index()
    conteos.columns = [var_cat, "Frecuencia"]

    fig = px.bar(
        conteos,
        x=var_cat,
        y="Frecuencia",
        title=f"Distribución de {var_cat}",
        color_discrete_sequence=["#ff7f0e"]
    )
    fig.update_layout(
        xaxis_title=var_cat,
        yaxis_title="Frecuencia",
        margin=dict(l=40, r=20, t=50, b=40),
        height=400
    )

    return dcc.Graph(figure=fig)

@app.callback(
    Output('eda2-num-output', 'children'),
    Input('eda2-num-dropdown', 'value')
)
def eda2_numerica(var_num):
    fig = px.histogram(
        df,
        x=var_num,
        nbins=20,
        title=f"Distribución de {var_num}",
        color_discrete_sequence=["#1f77b4"]
    )
    fig.update_layout(
        xaxis_title=var_num,
        yaxis_title="Frecuencia",
        margin=dict(l=40, r=20, t=50, b=40),
        height=400
    )

    return dcc.Graph(figure=fig)

#===================================================== CALLBACK MODELO
@app.callback(
    Output('visualizacion-modelo', 'children'),
    Input('modelo-dropdown', 'value')
)
def actualizar_visualizacion(nombre_modelo):
    modelo = final_models[nombre_modelo]
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    # === MATRIZ DE CONFUSIÓN ===
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicción", y="Valor Real", color="Cantidad"),
        x=["<=50K", ">50K"],
        y=["<=50K", ">50K"]
    )
    fig_cm.update_layout(
        title=f"Matriz de Confusión - {nombre_modelo}",
        height=500
    )

    # === CURVA ROC ===
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Azar', line=dict(dash='dash')))
    fig_roc.update_layout(
        title=f"Curva ROC - {nombre_modelo}",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        height=500
    )

    # === FEATURE IMPORTANCES (solo para Random Forest y XGBoost) ===
    feature_importances_content = []
    
    if nombre_modelo != "Regresión Logística":
        try:
            feature_names = modelo.named_steps["preprocessing"].get_feature_names_out()
            importances = modelo.named_steps["classifier"].feature_importances_

            # Ordenar las top 10
            idx = np.argsort(importances)[::-1][:10]
            df_imp = pd.DataFrame({
                "Variable": feature_names[idx],
                "Importancia": importances[idx]
            })

            fig_imp = px.bar(
                df_imp,
                x="Importancia",
                y="Variable",
                orientation='h',
                title=f"Top 10 Variables más Importantes - {nombre_modelo}",
                text="Importancia"
            )
            fig_imp.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=500
            )
            
            feature_importances_content = [
                # Fila 2: Feature Importances (ocupa toda la fila)
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_imp)
                    ], width=12)
                ])
            ]
            
        except Exception as e:
            fig_imp = go.Figure()
            fig_imp.add_annotation(text=f"No se pudieron obtener las feature importances ({str(e)})")
            fig_imp.update_layout(height=300)
            
            feature_importances_content = [
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_imp)
                    ], width=12)
                ])
            ]
    else:
        # Mensaje para Regresión Logística
        feature_importances_content = [
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Feature Importances no disponibles", 
                               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '20px'}),
                        html.P("La Regresión Logística no proporciona feature importances nativas.",
                              style={'textAlign': 'center', 'color': '#95a5a6', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
                ], width=12)
            ])
        ]

    return [
        html.H5(f"Visualización del Modelo - {nombre_modelo}", style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        # Fila 1: Matriz de Confusión y Curva ROC lado a lado
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_cm)
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=fig_roc)
            ], width=6)
        ], className="mb-4"),
        
        # Feature Importances (solo se muestra si no es Regresión Logística)
        *feature_importances_content
    ]


#=================================================CALL BACK INDICADORES
@app.callback(
    Output('indicadores-modelo', 'children'),
    Input('modelo-dropdown', 'value')
)
def actualizar_indicadores(nombre_modelo):
    modelo = final_models[nombre_modelo]
    y_pred = modelo.predict(X_test)

    # Calcular probas para AUC
    try:
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        else:
            y_proba = modelo.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
    except Exception:
        auc_score = None

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return html.Div([
        html.H5(f"Métricas de Evaluación - {nombre_modelo}", 
                style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
        
        dbc.Row([
            # Accuracy
            dbc.Col([
                html.Div([
                    html.H6('Accuracy', style={'color': '#27ae60', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H4(f'{accuracy:.4f}', 
                           style={'color': '#27ae60', 'textAlign': 'center', 'fontWeight': 'bold', 'margin': '10px 0'}),
                    html.P(f'{accuracy*100:.1f}%', 
                          style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#f0f8f4',
                    'borderRadius': '12px',
                    'border': '3px solid #27ae60',
                    'textAlign': 'center',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                })
            ], width=2),
            
            # Precision
            dbc.Col([
                html.Div([
                    html.H6('Precision', style={'color': '#3498db', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H4(f'{precision:.4f}', 
                           style={'color': '#3498db', 'textAlign': 'center', 'fontWeight': 'bold', 'margin': '10px 0'}),
                    html.P(f'{precision*100:.1f}%', 
                          style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#f0f8ff',
                    'borderRadius': '12px',
                    'border': '3px solid #3498db',
                    'textAlign': 'center',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                })
            ], width=2),
            
            # Recall
            dbc.Col([
                html.Div([
                    html.H6('Recall', style={'color': '#e74c3c', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H4(f'{recall:.4f}', 
                           style={'color': '#e74c3c', 'textAlign': 'center', 'fontWeight': 'bold', 'margin': '10px 0'}),
                    html.P(f'{recall*100:.1f}%', 
                          style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#fdf2f2',
                    'borderRadius': '12px',
                    'border': '3px solid #e74c3c',
                    'textAlign': 'center',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                })
            ], width=2),
            
            # F1-Score
            dbc.Col([
                html.Div([
                    html.H6('F1-Score', style={'color': '#9b59b6', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H4(f'{f1:.4f}', 
                           style={'color': '#9b59b6', 'textAlign': 'center', 'fontWeight': 'bold', 'margin': '10px 0'}),
                    html.P(f'{f1*100:.1f}%', 
                          style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#f8f0f8',
                    'borderRadius': '12px',
                    'border': '3px solid #9b59b6',
                    'textAlign': 'center',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                })
            ], width=2),
            
            # AUC-ROC (si está disponible)
            dbc.Col([
                html.Div([
                    html.H6('AUC-ROC', style={'color': '#f39c12', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H4(f'{auc_score:.4f}' if auc_score is not None else 'N/A', 
                           style={'color': '#f39c12', 'textAlign': 'center', 'fontWeight': 'bold', 'margin': '10px 0'}),
                    html.P(f'{auc_score*100:.1f}%' if auc_score is not None else 'No disponible', 
                          style={'textAlign': 'center', 'color': '#666', 'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '0'})
                ], style={
                    'padding': '25px',
                    'backgroundColor': '#fef9e7',
                    'borderRadius': '12px',
                    'border': '3px solid #f39c12',
                    'textAlign': 'center',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                })
            ], width=2) if auc_score is not None else html.Div()
            
        ], className="mb-4 justify-content-center"),
        
        # Información adicional
        html.Div([
            html.P(f"Modelo: {nombre_modelo}", 
                  style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#7f8c8d', 'marginTop': '20px'}),
            html.P(f"Total de predicciones: {len(y_pred)} | Positivos: {sum(y_pred)} | Negativos: {len(y_pred) - sum(y_pred)}",
                  style={'textAlign': 'center', 'fontSize': '14px', 'color': '#95a5a6'})
        ])
    ])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
