# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from sqlalchemy import create_engine
import shap
#from sources import *
import xgboost

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='input-cvr-state', type='text', value='12'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output-state'),
    html.Div(id='output-shap')
])




###
import shap
from shap.plots._force_matplotlib import draw_additive_plot

# ... class dashApp
# ... callback as method 
# matplotlib=False => retrun addaptativevisualizer, 
# if set to True the visualizer will render the result is the stdout directly
# x is index of wanted input
# class_1 is ma class to draw

import io
import base64

def figure_to_html_img(figure):
    """ figure to html base64 png image """ 
    try:
        tmpfile = io.BytesIO()
        figure.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        shap_html = html.Img(src=f"data:image/png;base64, {encoded}")
        return shap_html
    except AttributeError:
        return ""
###














@app.callback(Output('output-shap', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-cvr-state', 'value')])

def update_shap_figure(n_clicks, input_cvr):
    shap.initjs()

    # train XGBoost model
    X,y = shap.datasets.boston()

    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values(same syntax works for LightGBM, CatBoost, and scikit-learn models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print('value ', n_clicks)
    print('value ', type(n_clicks))
    print('value ', input_cvr)
    input_cvr = int(input_cvr)
    print('value ', type(input_cvr))
    
    #print(model.predict(X.iloc[12]))
    plt.style.use("_classic_test_patch") 
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values[1,:],
        X.iloc[input_cvr, :].drop(columns=["TARGET"], errors="ignore"),
        matplotlib=False
        )
    # set show=False to force the figure to be returned
    force_plot_mpl = draw_additive_plot(force_plot.data, (10, 4), show=False)
    return figure_to_html_img(force_plot_mpl)

    
    # visualize the first prediction's explanation

    #return(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])) # matplotlib=True

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader = False)