import os
from flask import Flask, render_template, request
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import shap
import plotly_express as px
global ff
ff = []
import queue

import logging
import threading
from threading import Thread

import time
import plotly.graph_objects as go # or plotly.express as px

import mpld3

fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(
        title=go.layout.Title(text="A Figure Specified By A Graph Object")
    )
)
import io
import base64
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







class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self):
        Thread.join(self)
        return self._return








app = Flask(__name__)




ee = dash.Dash(__name__,
    server=app,
    url_base_pathname='/ee/')
jj = dash.Dash(__name__,
    server=app,
    url_base_pathname='/jj/')
qq = dash.Dash(__name__,
    server=app,
    url_base_pathname='/qq/')

new_pca = dash.Dash(__name__,
    server=app,
    url_base_pathname='/new_pca/')


local_explain2 = dash.Dash(__name__,
    server=app,
    url_base_pathname='/local_explain2/')










table_plot = dash.Dash(__name__,
    server=app,
    url_base_pathname='/table_plot/')





SHAP_plot = dash.Dash(__name__,
    server=app,
    url_base_pathname='/SHAP_plot/')




what_plot = dash.Dash(__name__,
    server=app,
    url_base_pathname='/what_plot/')


pca_3_fig = dash.Dash(__name__,
    server=app,
    url_base_pathname='/pca_3_fig/')



tsne = dash.Dash(__name__,
    server=app,
    url_base_pathname='/tsne/')




dashboard_ji = dash.Dash(__name__,
    server=app,
    url_base_pathname='/dashboard_ji/')





dashboard_ji.layout= html.Div([dcc.Graph(figure=fig)])






tsne.layout= html.Div([dcc.Graph(figure=fig)])






pca_3_fig.layout= html.Div([dcc.Graph(figure=fig)])



what_plot.layout= html.Div([dcc.Graph(figure=fig)])



SHAP_plot.layout= html.Div([dcc.Graph(figure=fig)])






table_plot.layout= html.Div([dcc.Graph(figure=fig)])












local_explain2.layout= html.Div([dcc.Graph(figure=fig)])


new_pca.layout = html.Div([dcc.Graph(figure=fig)])

ee.layout = html.Div([dcc.Graph(figure=fig)])
jj.layout = html.Div([dcc.Graph(figure=fig)])
qq.layout = html.Div([dcc.Graph(figure=fig)])








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

@app.route("/data_explanation")
def data_explanation():
    return render_template("data_explanation.html")

def permutation_importance(model,X_data,y_data):
    perm = PermutationImportance(model).fit(X_data, y_data)
    PI = eli5.show_weights(perm, feature_names=X_data.columns.tolist())
    return PI

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


def _force_plot_html(*args):
    force_plot = shap.force_plot(*args, matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "400px", "border": 2})


def _force_plot_html2(*args):
    force_plot = shap.force_plot(*args, matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "400px", "border": 2})




@app.route("/ee")
def ind_cond_exp(model_line,X_train,y_data):
    
    
    
    empty_list = []

    for col in X_train.columns: 
        print(col) 
        empty_list.append(col)
    
    
    
    
    
    from pdpbox.pdp import pdp_interact, pdp_interact_plot
    X_features = empty_list

    features = empty_list[1:3]
    
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
    
    
    print(dropdown_selection)
    model_type = dropdown_selection[3]
    dropdown_selection = dropdown_selection[1]
    
    
    print('model type ji ',model_type)
    
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
        
        
        if 'RR' in model_type:
        
        
        
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
            
    
            return render_template('model_explanation_result.html',PI = PI,
                                   ICE = ICE , SH = "static/img/new_plot.png",
                                   SM = "static/img/new2_plot.png")
        
        
        if 'RF' in model_type:
            PI = permutation_importance(model,X_data,y_data)
            
            explainer = shap.TreeExplainer(model, X_data, feature_names=X_data.columns)
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
            
    
            return render_template('model_explanation_result.html',PI = PI,
                                   ICE = ICE , SH = "static/img/new_plot.png",
                                   SM = "static/img/new2_plot.png")
            
            
            
            
            
        if 'CC' in model_type:
            PI = permutation_importance(model,X_data,y_data)
            
            
            explainer = shap.KernelExplainer(model.predict_proba, X_data)
            shap_values = explainer.shap_values(X_data)
            
            
          
        
        
            shap.summary_plot(shap_values, X_data)
        
            import matplotlib.pyplot as pl
            pl.savefig('static/img/new_plot.png')
            pl.close()
        
            #ICE = ind_cond_exp(model,X_data,y_data)
            
            
            #global surgat
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.tree import plot_tree
            
            
            predictions = model.predict(X_data)
            
            
            
           
            
    
            return render_template('model_explanation_result_classification.html',PI = PI,
                                   SH = "static/img/new_plot.png"
                                   )
            
        
        
        
    if 'WI' in dropdown_selection:
        
        
       # print(res," resss")
        
       #
        import dash
        from dash.dependencies import Input, Output
        import dash_table
        import dash_core_components as dcc
        import dash_html_components as html
        
        app = dash.Dash(__name__)
        import pandas as pd
        #should be X data
        
        
        
        
            
        mean_list = []
        features = X_data.columns.tolist()
        for i in features:
            mean_list.append(round(X_data[i].mean()))
        
            
        explainer = shap.TreeExplainer(model)
        shap.initjs()
        
        params = features
        
        what_plot.layout = html.Div([
            dash_table.DataTable(
                id='table-editing-simple',
                columns=(
                    [{'id': 'Model', 'name': 'Model'}] +
                    [{'id': p, 'name': p} for p in params]
                ),
                data=[
                        
                        
                        dict(zip(features, mean_list)) 
                    #dict(Model=i, **{param: mean_list[i] for param in params})
                   # for i in range(0, len(mean_list))
                ],
                editable=True
            ),
            
            html.Div(id='datatable-interactivity-container')
        ])
        
        
        @what_plot.callback(
            Output('datatable-interactivity-container', "children"),
            Input('table-editing-simple', 'data'),
            Input('table-editing-simple', 'columns'))
        def update_graphs(rows, columns):
            df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
            print(rows)
            
            
            #
            rows = rows[0]
            col = []
            vvalue =[]
            for key in rows:
                print(key, '->', int(rows[key]))
                col.append(key)
                vvalue.append([int(rows[key])])
                
                
            ik=dict(zip(col,vvalue))
            instance = pd.DataFrame.from_dict(ik)
                        
                        
            print('instancceee ', instance)
            
           
            from shap.plots._force_matplotlib import draw_additive_plot

            
             
            # explain the model's predictions using SHAP values(same syntax works for LightGBM, CatBoost, and scikit-learn models)
            #explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance)
            shap.initjs()
            
            #plt.style.use("_classic_test_patch")
           
            ytu = model.predict(instance)
            print('ress ' , ytu)
            
            
            koko = _force_plot_html2(explainer.expected_value,
                                         shap_values,instance)
                                         
           
            #print('kkkk ',koko)
           
            print('Done')
            
            
            return koko
       #
       
       
       
       
       
       
       
       
       
       
       
       
       
        
        
        
        return render_template('local_explain_lime.html', LL = what_plot.index())
    
    
    if 'LL' in dropdown_selection:
        None
        #table and plots ========================================================
        import dash
        from dash.dependencies import Input, Output
        import dash_table
        import dash_core_components as dcc
        import dash_html_components as html
        import pandas as pd
        print('in LL')
        # make graph===============================================================
        table_plot.layout = html.Div([
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[
                    {"name": i, "id": i, "deletable": True, "selectable": True} for i in X_data.columns
                ],
                data=X_data.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="single",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= 10,
            ),
            html.Div(id='datatable-interactivity-container')
        ])
            
        print('miod LL')
     
            
            
            
            
            
            
        @table_plot.callback(
            Output('datatable-interactivity-container', "children"),
            Input('datatable-interactivity', "derived_virtual_data"),
            Input('datatable-interactivity', "derived_virtual_selected_rows"))
        def update_graphs(rows, derived_virtual_selected_rows):
            # When the table is first rendered, `derived_virtual_data` and
            # `derived_virtual_selected_rows` will be `None`. This is due to an
            # idiosyncrasy in Dash (unsupplied properties are always None and Dash
            # calls the dependent callbacks when the component is first rendered).
            # So, if `rows` is `None`, then the component was just rendered
            # and its value will be the same as the component's dataframe.
            # Instead of setting `None` in here, you could also set
            # `derived_virtual_data=df.to_rows('dict')` when you initialize
            # the component.
            if derived_virtual_selected_rows is None:
                derived_virtual_selected_rows = []
            
            dff = X_data if rows is None else pd.DataFrame(rows)
            
            colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
                      for i in range(len(dff))]
            
            print('my value',derived_virtual_selected_rows)
            print('i am row ', X_data.iloc[derived_virtual_selected_rows])
            print(type(derived_virtual_selected_rows))
            
            from shap.plots._force_matplotlib import draw_additive_plot

            
             
            ttt = X_data.loc[derived_virtual_selected_rows]
            # explain the model's predictions using SHAP values(same syntax works for LightGBM, CatBoost, and scikit-learn models)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(ttt)
            shap.initjs()
            
            plt.style.use("_classic_test_patch")
           
            
            
            
            bubu = _force_plot_html(explainer.expected_value,
                                         shap_values,ttt)
                                         
           
            
            shap_values = explainer.shap_values(X_data)
            #shap.force_plot(explainer.expected_value, shap_values, X_data)
            explain_all = _force_plot_html(explainer.expected_value,
                                         shap_values,X_data)
            
            
            
            print('bubu ',bubu)
            
            return bubu, explain_all
                    
                    
            
                
        return render_template('local_explain_lime.html', LL = table_plot.index())
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    if 'BD' in dropdown_selection:
        None
        
      #FI  
    if 'DB' in dropdown_selection:     
        
      #  if 'CC' in model_type:
         #   from explainerdashboard import ClassifierExplainer, ExplainerDashboard
          #  ExplainerDashboard(ClassifierExplainer(model, X_data, y_data)).run()
        
        if 'RF' in model_type:
            import threading
            import time
            
            def dashboard_exp(model, X_data, y_data):
                import dash_bootstrap_components as dbc

                from explainerdashboard import RegressionExplainer, ExplainerDashboard
                ExplainerDashboard(RegressionExplainer(model, X_data, y_data),bootstrap=dbc.themes.SANDSTONE,
                                   importances=True,
                        model_summary=False,
                        contributions=True,
                        
                        whatif=True,
                        shap_dependence=False,
                        shap_interaction=False,
                        decision_trees=False,
                        
                        hide_whatifindexselector=True,
                        hide_whatifprediction=True,
                        hide_inputeditor=False,
                        hide_whatifcontributiongraph=False, 
                        hide_whatifcontributiontable=True, 
                        hide_whatifpdp=False,
                        
                        hide_predindexselector=True, 
                        hide_predictionsummary=True,
                        hide_contributiongraph=False, 
                        hide_pdp=False, 
                        hide_contributiontable=True,
                        
                        hide_dropna=True,
                        hide_range=True,
                         hide_depth=True,
                         hide_sort=True,
                         hide_sample=True, # hide sample size input on pdp component
                        hide_gridlines=True, # hide gridlines on pdp component
                        hide_gridpoints=True,
                        
                        
                        
                         hide_cats_sort=True, # hide the sorting option for categorical features
                        hide_cutoff=True, # hide cutoff selector on classification components
                        hide_percentage=True, # hide percentage toggle on classificaiton components
                        hide_log_x=True, # hide x-axis logs toggle on regression plots
                        hide_log_y=True, # hide y-axis logs toggle on regression plots
                        hide_ratio=True, # hide the residuals type dropdown
                        hide_points=True, # hide the show violin scatter markers toggle
                        hide_winsor=True, # hide the winsorize input
                        hide_wizard=True, # hide the wizard toggle in lift curve component
                        hide_star_explanation=True,
                        
                        
                        ).run()
            
            t1 = threading.Thread(target=dashboard_exp, args=(model, X_data, y_data)) 
              
            
            t1.start() 
            
                      
                
                
                
            return '''<H2>
         Please follow this link <a href="http://localhost:8050/">LINK</a> 
      </H2>'''
                #dashboard_ji
            #app = db.flask_server()
            
            
            
        
        
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#-----------------------------------------------------------------------------------------------------------



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
    print(list(X_data.columns))
    
    
    series = pd.Series(hh) 
    
    
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    
    
    # load JS visualization code to notebook
    shap.initjs()
    
    #plt.style.use("_classic_test_patch")  
    #plt.clf()
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    #shap.force_plot(explainer.expected_value, shap_values[1,:], series, matplotlib=True, figsize=(22, 4))
    #shap.force_plot(explainer.expected_value, shap_values[10,:],  \
    #                series,feature_names=X_data.columns,\
     #               matplotlib=True, show=False)   
    
    
    
    
   # plt.savefig("gg.png",dpi=150, bbox_inches='tight')
        
    #yyy = shap.getjs()
    '''
    oo = yyy.matplotlib
    p = yyy.html  
    yyy_str = mpld3.fig_to_html(p)  
    print('dfsdfsdf ',p)     
    '''
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
    
    exp = explainer.explain_instance(series, model.predict )
                                     
    print(exp.local_pred)
    
    fig = exp.as_pyplot_figure(label = list(X_data.columns))
    
    fig_2 = exp.as_html(labels = list(X_data.columns))
    #print('dddd ',fig_2)
    
    plt.tight_layout()
    #fig = plt.figure(figsize = (18,8))
    
#    plt.tight_layout()
#    #plt.boxplot(bank_data.transpose())
#
#    #Add titles to the chart and axes
#    plt.hist(bank_data.transpose(), bins = 50)
#    plt.title('Boxplot of Bank Stock Prices (5Y Lookback)')
#    plt.xlabel('Bank')
#    plt.ylabel('Stock Prices')
#        
    #mpld3.show(fig)
    #
    html_str = mpld3.fig_to_html(fig)
    Html_file= open("templates/lime.html","w")
    Html_file.write(html_str)
    Html_file.close()
    #
    
    
    
    
    
   # plt.savefig('static/img/new34_plot.png')
    #plt.close()
    
       
        
    
    return render_template('local_result.html', LIME = html_str, SH = fig_2, gh = html_str)
















def thread_function(data1, data, yw):
    print("pca ki jai")
    import plotly_express as px
    import dash
    import dash_html_components as html
    import dash_core_components as dcc
    from dash.dependencies import Input, Output
    
    
    wwww = dash.Dash(__name__,
    server=app,
    url_base_pathname='/wwww/')






    fig = go.Figure(
        data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
        layout=go.Layout(
            title=go.layout.Title(text="A Figure Specified By A Graph Object")
        )
    )


    wwww.layout = html.Div([dcc.Graph(figure=fig)])
    
    
    
            
   # tips = px.data.tips()
    col_options = [dict(label=x, value=x) for x in data1.columns]
    dimensions = ['color']
            
            
    
    
    
    
    
    
    ###pca
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    
    import plotly.express as px
    from sklearn.decomposition import PCA
    
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    
    pca3 = PCA(n_components=3)
    components_3 = pca3.fit_transform(scaled_data)
    
    total_var = pca.explained_variance_ratio_.sum() * 100
    
    fig_3 = px.scatter_3d(
        components_3, x=0, y=1, z=2, color=yw,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    
    fig_3.show()
    #need to upload csv only
    
    fig = px.scatter(components, x=0, y=1, color=yw)
    fig.show()
    ###
    
    wwww.layout = html.Div(
        [
            html.H1("Demo"),
            html.Div(
                [
                    html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                    for d in dimensions
                ],
                style={"width": "25%", "float": "left"},
            ),
            dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
        ]
    )
    
    print('dimsum ', dimensions)
    @wwww.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
    def make_figure( color):
        print('ccc ',color)
        if color == None:
            my_color = None
        else:
            my_color = data1[color]
        return px.scatter(
            components,
            x = 0,
            y = 1,
            color=my_color,
            
            height=700,
        )

    
    
    
    
    
   # jj.layout = html.Div([dcc.Graph(figure=fig)])
    #www.layout = html.Div([dcc.Graph(figure=fig_3)])
    print("done")
    
    return wwww.index()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

   
    
    
    
    
    
    
    
    



#data explanation----------------------
@app.route("/upload_3", methods=['POST'])
def upload_3():
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
    
    import pandas as pd
    print('raJA ',ff)
    import warnings
    warnings.filterwarnings("ignore")

    data1 = pd.read_csv(ff[0])
    #print('datagg ',data1)
   
    
#    dim = data = data1[['bedrooms', 'bathrooms', 'sqft_living',
#       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
#       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
#       'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']]
    
        
    if 'PCA' in dropdown_selection:
        input_list = request.form.to_dict()
    
    
    
        #data['some_key'] = "Some Value"
        print('input values ', input_list)
        print('input values ', type(input_list))
        
        
        
        target_name = input_list['lname']
        
        
        
        target_name = target_name.split("'")[1]
        print('taget ss ',target_name)
        print('taget ss ',type(target_name))
        
        
        feature_name = input_list['features']
        feature_name = feature_name.split(",")
        uuu=[]
        for i in range(0,len(feature_name)):
            uuu.append(feature_name[i].split("'")[1])
            
        print('nuna yadav ', uuu)
        
        data = data1[uuu]
        yw = data1[target_name]
    
        
        #twrv = ThreadWithReturnValue(target=thread_function, args=(data1,data,yw))
        #twrv.start()
        #value = twrv.join()
        #data_explanation_thread = threading.Thread()
        #data_explanation_thread.start()
        #value = data_explanation_thread.join()
        #que = queue.Queue()
        #value = que.get()
        #print(value)
        #value = thread_function(data1,data,yw)
        
        print("pca ki jai")
        import plotly_express as px
        import dash
        import dash_html_components as html
        import dash_core_components as dcc
        from dash.dependencies import Input, Output
                
       # tips = px.data.tips()
        col_options = [dict(label=x, value=x) for x in data1.columns]
        dimensions = ['Select dimension to be shown in colour']
                
                
        
        
        
        
        
        
        ###pca
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        
        import plotly.express as px
        from sklearn.decomposition import PCA
        
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(data)
        
        pca3 = PCA(n_components=3)
        components_3 = pca3.fit_transform(scaled_data)
        
        total_var = pca.explained_variance_ratio_.sum() * 100
        
        fig_3 = px.scatter_3d(
            components_3, x=0, y=1, z=2, color=yw,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        
        fig_3.show()
        #need to upload csv only
        
        fig = px.scatter(components, x=0, y=1, color=yw)
        fig.show()
        ###
        
        new_pca.layout = html.Div(
            [
                html.H1("Demo"),
                html.Div(
                    [
                        html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                        for d in dimensions
                    ],
                    style={"width": "25%", "float": "left"},
                ),
                dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
                                html.H3("Principal Component Analysis (PCA) is an unsupervised linear transformation technique that is widely used across different fields, most prominently for feature extraction and dimensionality reduction. Other popular applications of PCA include exploratory data analyses and de-noising of signals in stock market trading, and the analysis of genome data and gene expression levels in the field of bioinformatics."),

            ]
        )
        
        print('dimsum ', dimensions)
        @new_pca.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
        def make_figure( color):
            print('ccc ',color)
            if color == None:
                my_color = None
            else:
                my_color = data1[color]
            return px.scatter(
                components,
                x = 0,
                y = 1,
                color=my_color,
                
                height=700,
            )

        
        
        
        
        
        jj.layout = html.Div([dcc.Graph(figure=fig)])
        qq.layout = html.Div([dcc.Graph(figure=fig_3)])
        print("done")
        
        

    
        

        
        
        
        
        
        return render_template('pca_result.html',PCA = new_pca.index(), PCAA = new_pca.index())
    
    
    
    #######
    #p3
    if 'P3' in dropdown_selection:
        input_list = request.form.to_dict()
    
    
    
        #data['some_key'] = "Some Value"
        print('input values ', input_list)
        print('input values ', type(input_list))
        
        
        
        target_name = input_list['lname']
        
        
        
        target_name = target_name.split("'")[1]
        print('taget ss ',target_name)
        print('taget ss ',type(target_name))
        
        
        feature_name = input_list['features']
        feature_name = feature_name.split(",")
        uuu=[]
        for i in range(0,len(feature_name)):
            uuu.append(feature_name[i].split("'")[1])
            
        print('nuna yadav ', uuu)
        
        data = data1[uuu]
        yw = data1[target_name]
    
        
        #twrv = ThreadWithReturnValue(target=thread_function, args=(data1,data,yw))
        #twrv.start()
        #value = twrv.join()
        #data_explanation_thread = threading.Thread()
        #data_explanation_thread.start()
        #value = data_explanation_thread.join()
        #que = queue.Queue()
        #value = que.get()
        #print(value)
        #value = thread_function(data1,data,yw)
        
        print("pca ki jai")
        import plotly_express as px
        import dash
        import dash_html_components as html
        import dash_core_components as dcc
        from dash.dependencies import Input, Output
                
       # tips = px.data.tips()
        col_options = [dict(label=x, value=x) for x in data1.columns]
        dimensions = ['Select dimension to be shown in colour']
                
                
        
        
        
        
        
        
        ###pca
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        
        import plotly.express as px
        from sklearn.decomposition import PCA
        
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        
        pca3 = PCA(n_components=3)
        components_3 = pca3.fit_transform(data)
        
        total_var = pca.explained_variance_ratio_.sum() * 100
        
        fig_3 = px.scatter_3d(
            components_3, x=0, y=1, z=2, color=yw,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        
        fig_3.show()
        #need to upload csv only
        
        fig = px.scatter(components, x=0, y=1, color=yw)
        fig.show()
        ###
        
        pca_3_fig.layout = html.Div(
            [
                html.H1("Demo"),
                html.Div(
                    [
                        html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                        for d in dimensions
                    ],
                    style={"width": "25%", "float": "left"},
                ),
                dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
                html.H3("Principal Component Analysis (PCA) is an unsupervised linear transformation technique that is widely used across different fields, most prominently for feature extraction and dimensionality reduction. Other popular applications of PCA include exploratory data analyses and de-noising of signals in stock market trading, and the analysis of genome data and gene expression levels in the field of bioinformatics."),
            ]
        )
        
        print('dimsum ', dimensions)
        @pca_3_fig.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
        def make_figure( color):
            print('ccc ',color)
            if color == None:
                my_color = None
            else:
                my_color = data1[color]
            return px.scatter_3d(
                components_3,
                x = 0,
                y = 1,
                z=2,
                color=my_color,
                
                height=700,
            )

        
        
        
        
        
        jj.layout = html.Div([dcc.Graph(figure=fig)])
        qq.layout = html.Div([dcc.Graph(figure=fig_3)])
        print("done")
        
        

    
        

        
        
        
        
        
        return render_template('pca_result.html',PCA = pca_3_fig.index(), PCAA = pca_3_fig.index())
    
    
    
    
    #######
    
    
    
    
    
    
    
    
    
    #t3
    
    if 'TSNE' in dropdown_selection:
        None
        input_list = request.form.to_dict()
    
    
    
        #data['some_key'] = "Some Value"
        print('input values ', input_list)
        print('input values ', type(input_list))
        
        
        
        target_name = input_list['lname']
        
        
        
        target_name = target_name.split("'")[1]
        print('taget ss ',target_name)
        print('taget ss ',type(target_name))
        
        
        feature_name = input_list['features']
        feature_name = feature_name.split(",")
        uuu=[]
        for i in range(0,len(feature_name)):
            uuu.append(feature_name[i].split("'")[1])
            
        print('nuna yadav ', uuu)
        
        data = data1[uuu]
        yw = data1[target_name]
    
        
        #twrv = ThreadWithReturnValue(target=thread_function, args=(data1,data,yw))
        #twrv.start()
        #value = twrv.join()
        #data_explanation_thread = threading.Thread()
        #data_explanation_thread.start()
        #value = data_explanation_thread.join()
        #que = queue.Queue()
        #value = que.get()
        #print(value)
        #value = thread_function(data1,data,yw)
        
        print("pca ki jai")
        import plotly_express as px
        import dash
        import dash_html_components as html
        import dash_core_components as dcc
        from dash.dependencies import Input, Output
                
       # tips = px.data.tips()
        col_options = [dict(label=x, value=x) for x in data1.columns]
        dimensions = ['Select dimension to be shown in colour']
                
                
        
        
        
        
        
        
        ###pca
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        
        from sklearn.manifold import TSNE
        import plotly.express as px


        tsne_algo = TSNE(n_components=3, random_state=0)
        projections = tsne_algo.fit_transform(data, )
        
        
        
        fig_3 = px.scatter_3d(
            projections, x=0, y=1, z=2, color=yw,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        
        fig_3.show()
        #need to upload csv only
        
        
        
        tsne.layout = html.Div(
            [
                html.H1("Demo"),
                html.Div(
                    [
                        html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                        for d in dimensions
                    ],
                    style={"width": "25%", "float": "left"},
                ),
                dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
                html.H3("t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. It was developed by Laurens van der Maatens and Geoffrey Hinton in 2008."),
            ]
        )
        
        print('dimsum ', dimensions)
        @tsne.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
        def make_figure( color):
            print('ccc ',color)
            if color == None:
                my_color = None
            else:
                my_color = data1[color]
            return px.scatter_3d(
                projections,
                x = 0,
                y = 1,
                z=2,
                color=my_color,
                
                height=700,
            )

        
        
        
        
       
        
        

    
        

        
        
        
        
        
        return render_template('pca_result.html',PCA = tsne.index(), PCAA = tsne.index())
        
        
        
        
        
        
            
        
        
    
    if 'PP' in dropdown_selection:
        import plotly_express as px
        import dash
        import dash_html_components as html
        import dash_core_components as dcc
        from dash.dependencies import Input, Output
        
        col_options = [dict(label=x, value=x) for x in data1.columns]
        dimensions = ["x", "y", "color", "facet_col", "facet_row"]
        
      
        
        local_explain2.layout = html.Div(
            [
                html.H1("dashboard"),
                html.Div(
                    [
                        html.P([d + ":", dcc.Dropdown(id=d, options=col_options)])
                        for d in dimensions
                    ],
                    style={"width": "25%", "float": "left"},
                ),
                dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
            ]
        )
        
        
        @local_explain2.callback(Output("graph", "figure"), [Input(d, "value") for d in dimensions])
        def make_figure(x, y, color, facet_col, facet_row):
            if x == None:
                x = data1[data1.columns[2]]
            if y == None:
                y = data1[data1.columns[3]]
                
            return px.scatter(
                data1,
                x=x,
                y=y,
                color=color,
                facet_col=facet_col,
                facet_row=facet_row,
                height=700,
            )
        

        
        
        

    
        

        
        
        
        
        
        return render_template('local_local_result.html',LL = local_explain2.index(), TA = 1)
        
        
    if 'DE' in dropdown_selection:
        None
        































if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    #app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


