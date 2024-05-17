import argparse

from webapp.app import app
from webapp.pages import home


server = app.server
app.layout = home.index_page


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int,
        help="port for webserver",
        default=8050
    )
    args = parser.parse_args()
    app.run_server(host="0.0.0.0", port=args.port, debug=True)
