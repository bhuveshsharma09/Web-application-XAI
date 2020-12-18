import os
from flask import Flask, render_template, request
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import shap
global ff
ff = []

import plotly.graph_objects as go # or plotly.express as px


fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(
        title=go.layout.Title(text="A Figure Specified By A Graph Object")
    )
)

import dash
import dash_core_components as dcc
import dash_html_components as html








import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance

import pycebox.ice as pice 
import base64
from io import BytesIO


app = Flask(__name__)




ee = dash.Dash(__name__,
    server=app,
    url_base_pathname='/ee/')
ee.layout = html.Div([dcc.Graph(figure=fig)])








# adding dummy comment
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/model_explanation")
def model_explanation():
    return render_template("model_explanation.html")

def permutation_importance(model,X_data,y_data):
    perm = PermutationImportance(model).fit(X_data, y_data)
    PI = eli5.show_weights(perm, feature_names=X_data.columns.tolist())
    return PI

@app.route("/ee")
def ind_cond_exp(model_line,X_train,y_data):
    
    from pdpbox.pdp import pdp_interact, pdp_interact_plot
    X_features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']

    features = ['sqft_living', 'bathrooms']
    
    interaction = pdp_interact(
      model=model_line,
      dataset=X_train,
      model_features=X_features,
      features=features 
    )
    
    
    #pdp_goals = pdp.pdp_isolate(model=model_line, dataset=X_train, model_features=X_features, feature='sqft_living')
    
    
    
    pdp_interact_plot(interaction, plot_type='grid', feature_names=features);
    
    
    
    import seaborn as sns
    
    pdp = interaction.pdp.pivot_table(
        values='preds', 
        columns=features[0], 
        index=features[1]
    )[::-1] # Slice notation to reverse index order so y axis is ascending
    
    #plt.figure(figsize=(10,8))
  #  sns.heatmap(pdp, annot=True, fmt='.2f', cmap='viridis')
  #  plt.title('Partial Dependence on Interest Rate on Annual Income & Credit Score');
    
    
    #import plotly.graph_objs as go
    
    surface = go.Surface(x=pdp.columns, 
                         y=pdp.index, 
                         z=pdp.values)
    
    fig = go.Figure(surface)
    fig.show()
    ee.layout = html.Div([dcc.Graph(figure=fig)])
    print("done")
    

    
    return ee.index()
     
    
        
    
@app.route("/upload", methods=['POST'])
def upload():
    print('eer  0', request.form)
    dropdown_selection = str(request.form)
    dropdown_selection = dropdown_selection.split()
    dropdown_selection = dropdown_selection[1]
    
    print(dropdown_selection, "  nuna bhai")
    
    
    target = 'images/'
    print('tt' , target)

    if not os.path.isdir(target):
        os.mkdir(target)
    global ff
    ff= []
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print('des',destination)
        file.save(destination)
        ff.append(destination)
   
        
    mypath = os. getcwd()
    onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    

    print('raJA ',ff)
    import warnings
    warnings.filterwarnings("ignore")


    with open(ff[0], 'rb') as file:
        model = pickle.load(file)

    with open(ff[1], 'rb') as file:
         X_data = pickle.load(file)

    with open(ff[2], 'rb') as file:
        y_data = pickle.load(file)
        
    if 'GL' in dropdown_selection:
        
        PI = permutation_importance(model,X_data,y_data)
        
        
        
        
        row_to_show = 5
    
        data_for_prediction = X_data.iloc[row_to_show]
        
        
        explainer = shap.Explainer(model, X_data, feature_names=X_data.columns)
        shap_values = explainer.shap_values(X_data)
    
    
        shap.summary_plot(shap_values, X_data)
    
        import matplotlib.pyplot as pl
        pl.savefig('static/img/new_plot.png')
        pl.close()
    
        ICE = ind_cond_exp(model,X_data,y_data)
        
        
        
        #global surgat
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.tree import plot_tree
        
        
        predictions = model.predict(X_data)
        dt = DecisionTreeRegressor(random_state = 100, max_depth=3)
        # We fit the shallow tree to the matrix X and the predictions of the random forest model 
        dt.fit(X_data, predictions)
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        
        plot_tree(dt, feature_names=list(X_data.columns), precision=3, 
                   filled=True, fontsize=12, impurity=True)
        pl.savefig('static/img/new2_plot.png')
        pl.close()
        
        #dt.score(X_test, predictions)
        
        
        
        
        
        
        
        
        
        
        return render_template('model_explanation_result.html',PI = PI, ICE = ICE , SH = "static/img/new_plot.png", SM = "static/img/new2_plot.png")
    else:
        
        mean_list = []
        features = X_data.columns.tolist()
        for i in features:
            mean_list.append(round(X_data[i].mean()))
        
        res = dict(zip(features, mean_list)) 
       # print(res," resss")
        
        
        
        
        return render_template('local_explanation_result-1.html', res = res)




@app.route("/upload2", methods=['POST'])
def upload2():
    from werkzeug.datastructures import ImmutableMultiDict
    
    with open(ff[0], 'rb') as file:
        model = pickle.load(file)

    with open(ff[1], 'rb') as file:
        X_data = pickle.load(file)

    with open(ff[2], 'rb') as file:
        y_data = pickle.load(file)
    
    
    
    
    print('start')
    print(request.form)
    hh = request.form
    hh = hh.to_dict(flat=False)
    print('hh ',hh)
    for file in request.files.getlist("gg"):
        print(file)
    print('end')
    
    
    series = pd.Series(hh) 
    
    
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    
    
    # load JS visualization code to notebook
    shap.initjs()
    
    plt.style.use("_classic_test_patch")  
    
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    #shap.force_plot(explainer.expected_value, shap_values[1,:], series, matplotlib=True, figsize=(22, 4))
    shap.force_plot(explainer.expected_value, shap_values[10,:], series)             
    series = series.tolist()
    print('im a he ',series)
    pp = []
    for i in series:
        for j in i:
            j = float(j)
            pp.append(j)

    
    
    
    
    
    
    series = np.array(pp)
    print('im a she ',series)
    
    
    
    
    
    
    
    
    #lime
    import lime
    from lime.lime_tabular import LimeTabularExplainer 
    explainer = LimeTabularExplainer(X_data, mode='regression', 
                                 feature_names=list(X_data.columns), 
                                 random_state=42, 
                                 discretize_continuous=False,
                                 kernel_width=0.2) 
    
    exp = explainer.explain_instance(series, model.predict, )
                                     
    print(exp.local_pred)
    
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('static/img/new34_plot.png')
    plt.close()
    
       
        
    
    return render_template('local_result.html', LIME = "static/img/new34_plot.png")



if __name__ == "__main__":
    app.run()
    #app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


