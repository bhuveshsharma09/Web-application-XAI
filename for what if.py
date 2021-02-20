import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
import pandas as pd
#should be X data
df = pd.read_csv('iris_csv.csv')




mean_list = []
features = df.columns.tolist()
features.remove('class')
for i in features:
    mean_list.append(round(df[i].mean()))
    print(i)

params = features

app.layout = html.Div([
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
    dcc.Graph(id='table-editing-simple-output')
])


@app.callback(
    Output('table-editing-simple-output', 'figure'),
    Input('table-editing-simple', 'data'),
    Input('table-editing-simple', 'columns'))
def display_output(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    print(rows)
    
    
    return {
        'data': [{
            'type': 'parcoords',
            'dimensions': [{
                'label': col['name'],
                'values': df[col['id']]
            } for col in columns]
        }]
    }


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader = False)