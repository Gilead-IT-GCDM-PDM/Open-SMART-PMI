# --- Imports

# External packages
import dash
from dash import dcc
from dash import html
# import pandas as pd
from dash.exceptions import PreventUpdate


# App modules
import utilities
from webapp import about_app
from webapp.app import app
# import parser
import predict


# --- App functions

upload_file_sdf = dcc.Upload(
    id='upload-data-sdf',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    style={
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'display': 'inline-block',
    },
    multiple=True
)


@app.callback(
    dash.dependencies.Output('attributes', 'children'),
    [dash.dependencies.Input('upload-data-sdf', 'contents')],
    [dash.dependencies.State('upload-data-sdf', 'filename')]
)
def compute_pmi(contents, names):
    if contents is not None:
        df = utilities.read_sdf_data(contents, names)
        estimates = predict.make_predictions(df)

        cols = ['NAME', 'SMART-PMI', 'COMPLEXITY', 'MW', 'SMILES', 'FILENAME']
        display = estimates[cols]

        output = dash.dash_table.DataTable(
            display.to_dict('records'),
            [{"name": i, "id": i} for i in display.columns]
        )
        return output
    raise PreventUpdate


index_page = html.Div(children=[
    about_app.HEADER,
    html.P(children=about_app.INSTRUCTIONS),
    html.H4('Please upload Compound Specification in SDF format', style={"fontWeight": "bold"}),
    upload_file_sdf,
    html.Label(id='filename-sdf'),

    html.Br(),
    dcc.Loading([
        html.Div(id='attributes'),
        html.Div(children=[html.Br()] * 7)
    ])
])
