from dash import html


HEADER = html.H2("Smart-PMI Estimator",
                 style={'color': 'rgb(240, 50, 50)'})


ABOUT = [
    html.Label("This app is to estimate Smart-PMI from molecule's sdf specification ")
]


INSTRUCTIONS = [
    html.Label('1. Please select/drag drop molecule specification in SDF format'),
    html.Label('2. Once selected, please wait for the calculation to complete',
               'This typically takes only a few seconds'),
    html.Label('3. Once processing completes, the results will appears in the table below')
]
