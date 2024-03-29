# --- Imports

# Standard library
import os
import base64
# import subprocess
import time
# import traceback

# External packages
import dash
from dash import dcc
from dash import html
# import pandas as pd
from dash.exceptions import PreventUpdate


# App modules
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
    multiple=False
)


@app.callback(
    dash.dependencies.Output('attributes', 'children'),
    [dash.dependencies.Input('upload-data-sdf', 'contents')],
    [dash.dependencies.State('upload-data-sdf', 'filename')]
)
def compute_pmi(content, name):
    if content is not None and 'sdf' in name.lower():
        df = read_sdf_data(content, name)
        estimates = predict.make_predictions(df)

        complexity = estimates.iloc[0]['molComplexity']
        pmi = estimates.iloc[0]['SMART-PMI']
        mw = estimates.iloc[0]['molwt']

        cols = ['ID', 'SMILES', 'molComplexity', 'molwt', 'SMART-PMI']
        display = estimates[cols]

        attrs = html.Div([
            html.H5(f'Molecular Complexity: {complexity}'),
            html.H5(f'SMART-PMI: {pmi}'),
            html.H5(f'Molecular Weight: {mw}'),
            html.Br(),

            html.H5('Attributes'),
            dash.dash_table.DataTable(
                display.to_dict('records'),
                [{"name": i, "id": i} for i in display.columns]
            )
        ])

        return attrs
    raise PreventUpdate


def persist_sdf_data(content, name):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    with open('tmp/uploaded.sdf', 'wb') as f:
        f.write(decoded)
    return decoded


def read_sdf_data(content, name):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    os.makedirs('tmp', exist_ok=True)
    random_filename = f'tmp/{int(time.monotonic()*1000)}.sdf'
    with open(random_filename, 'wb') as f:
        f.write(decoded)

    df = predict.read_sdf_files([random_filename])
    df['ID'] = name

    # os.remove(random_filename)

    return df



def compute_descriptors():
    # subprocess.run(["bin/compoundcomplexity.sh", "-sdf", "sample/uploaded.sdf"], env=os.environ, check=True)
    # df = parser.parse('descriptors.desc')
    df = None
    return df


index_page = html.Div([
    html.Table([
        html.Tr([
            html.Td(about_app.HEADER),
            html.Td(html.P(children=about_app.INSTRUCTIONS)),
        ]),
    ]),
    html.Table([
        html.Tr([
            html.Td(html.H4('Please upload Compound Specification in SDF format', style={"font-weight": "bold"})),
        ]),
        html.Tr([
            html.Td(upload_file_sdf),
        ]),
        html.Tr([
            html.Td(html.Label(id='filename-sdf')),
        ]),
    ]),

    dcc.Loading([
        html.Div(id='attributes'),
        html.Div(children=[html.Br()] * 1)
    ])
])
