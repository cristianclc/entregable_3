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
app.title = "Dashboard del Proyecto Final "
server = app.server  

#============================================================

final_models = joblib.load("models.joblib") #carga modelos

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

df = pd.read_csv("adult.csv")
df.columns = columns

#==============================================================

var_no = ["native.country", "income", "education.num", "fnlwgt"]

binarias = [col for col in df.columns if df[col].nunique() == 2 and col != "income"]
numericas = [col for col in df.columns if df[col].dtype == "int64" and col not in var_no]
categoricas = [col for col in df.columns if col not in (numericas+binarias) and col not in var_no]

#==============================================================

X = df[binarias+categoricas+numericas]
y = df["income"]

map_bin = {
    ' >50K': 1, 
    ' <=50K': 0, 
}

y = y.replace(map_bin)

#=============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#=============================================================

subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.Ul([
            html.Li('Tipo de problema: clasificación / regresión / agrupamiento / series de tiempo'),
            html.Li('Variable objetivo o de interés: Nombre de la variable')
        ])
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.Ul([
            html.Li('Limpieza y transformación de datos'),
            html.Li('División del dataset en entrenamiento y prueba o validación cruzada')
        ])
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.Ul([
            html.Li('Modelo(s) seleccionados'),
            html.Li('Justificación de la elección'),
            html.Li('Ecuación o representación matemática si aplica')
        ])
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento'),
            html.Li('Métricas de evaluación: RMSE, MAE, Accuracy, etc.'),
            html.Li('Validación utilizada')
        ])
    ])
])


subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. Análisis Univariado', children=[
        html.H4('a. Análisis univariado'),
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
    ]),

    dcc.Tab(label='b. Análisis Bivariado', children=[
        html.H4('b. Análisis Bivariado - Relaciones con variable objetivo'),
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
    ]),

    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),
        dcc.Dropdown(
            id='modelo-dropdown',
            options=[{'label': k, 'value': k} for k in final_models.keys()],
            value=list(final_models.keys())[0],
            clearable=False
        ),
        html.Div(id='visualizacion-modelo')
        ]),

    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('d. Indicadores de Evaluación del Modelo'),
        html.Div(id='indicadores-modelo')
        ]),
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('e. Limitaciones y Consideraciones Finales'),
        html.Ul([
            html.Li('Restricciones del análisis'),
            html.Li('Posibles mejoras futuras')
        ])
    ])
])


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
        html.H2('Objetivos y Justificación'),
        html.H4('Objetivo General'),
        html.Ul([html.Li('Objetivo general del proyecto')]),
        html.H4('Objetivos Específicos'),
        html.Ul([
            html.Li('Objetivo específico 1'),
            html.Li('Objetivo específico 2'),
            html.Li('Objetivo específico 3')
        ]),
        html.H4('Justificación'),
        html.P('Explicación breve sobre la importancia de abordar el problema planteado y los beneficios esperados.')
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        html.P('Resumen de conceptos teóricos (definiciones formales) claves relacionados con el proyecto. Se pueden incluir referencias o citas.')
    ]),
    dcc.Tab(label='6. Metodología', children=[
        html.H2('Metodología'),
        subtabs_metodologia
    ]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2('Resultados y Análisis Final'),
        subtabs_resultados
    ]),
    dcc.Tab(label='8. Conclusiones', children=[
        html.H2('Conclusiones'),
        html.Ul([
            html.Li('Listar los principales hallazgos del proyecto'),
            html.Li('Relevancia de los resultados obtenidos'),
            html.Li('Aplicaciones futuras y recomendaciones')
        ])
    ])
]

app.layout = dbc.Container([
    html.H1("Dashboard del Proyecto Final", className="text-center my-4"),
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
    fig_cm.update_layout(title=f"Matriz de Confusión - {nombre_modelo}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Azar', line=dict(dash='dash')))
    fig_roc.update_layout(
        title=f"Curva ROC - {nombre_modelo}",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)"
    )

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
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    except Exception as e:
        fig_imp = go.Figure()
        fig_imp.add_annotation(text=f"No se pudieron obtener las feature importances ({str(e)})")

    return [
        html.H5(f"Visualización del Modelo - {nombre_modelo}"),
        dcc.Graph(figure=fig_cm),
        dcc.Graph(figure=fig_roc),
        dcc.Graph(figure=fig_imp)
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

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    if auc_score is not None:
        metrics["AUC-ROC"] = auc_score

    df_metrics = pd.DataFrame(metrics, index=[nombre_modelo]).T.reset_index()
    df_metrics.columns = ["Métrica", "Valor"]

    table = dash_table.DataTable(
        data=df_metrics.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_metrics.columns],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'column_id': 'Valor'}, 'backgroundColor': 'rgba(0,123,255,0.05)'}
        ]
    )

    return html.Div([
        html.H5(f"Métricas de Evaluación - {nombre_modelo}"),
        table
    ])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
