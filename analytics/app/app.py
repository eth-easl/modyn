from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import Dash, dcc, html

# -------------------------------------------------------------------------------------------------------------------- #
#                                                         STYLE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

theme = {
    "dark": False,
}

stylesheets = {"mantine": "https://unpkg.com/@mantine/core@7/styles.css", "dbc_flatly": dbc.themes.FLATLY}
dash._dash_renderer._set_react_version("18.2.0")

pio.templates.default = "plotly"
# pio.templates.default = "plotly_white"
# pio.templates.default = "plotly_dark"
# pio.templates.default = "ggplot2"
# pio.templates.default = "seaborn"
# pio.templates.default = "simple_white"
# pio.templates.default = "presentation"
# pio.templates.default = "xgridoff"
# pio.templates.default = "ygridoff"
# pio.templates.default = "gridon"
# pio.templates.default = "none"

# -------------------------------------------------------------------------------------------------------------------- #
#                                                          APP                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

name = "Modyn Pipeline Evaluation"
app = Dash(
    name,
    title=name,
    use_pages=True,
    pages_folder=Path(__file__).parent / "pages",
    external_stylesheets=[stylesheets["dbc_flatly"]],
)

app.layout = html.Div(
    [
        html.H1("Modyn"),
        html.Div(
            [
                html.Div(dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"]))
                for page in dash.page_registry.values()
            ]
        ),
        dash.page_container,
    ],
    style={"padding": "50px 100px"},
)


def main() -> None:
    app.run(debug=True, port=12312)


if __name__ == "__main__":
    main()
