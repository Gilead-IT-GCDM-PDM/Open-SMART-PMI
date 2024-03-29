import dash
# from dash import dcc
# from dash import html
# import dash_bootstrap_components as dbc
# import dash_design_kit as ddk

from webapp.app import app
from webapp.pages import home

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(stream=sys.stdout))

server = app.server
# print("Dash Core Version: ", dcc.__version__)   # 0.6.0 or above is required


app.layout = home.index_page


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port="8050", debug=True)
