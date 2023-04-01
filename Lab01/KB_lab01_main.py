from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np
import random

app = Dash(__name__)

app.layout = html.Div([

    dcc.Graph(id='indicator-graphic'),
    dcc.Markdown("""
                Number of Samples
            """),
    dcc.Slider(
        1,
        100,
        step=10,
        id='size-slider',
        value=30,
    ),
    dcc.Markdown("""
                Number of Nodes of class 1
            """),
    dcc.Slider(
        1,
        3,
        step=1,
        id='node1-slider',
        value=1,
    ),
    dcc.Markdown("""
                Number of Nodes of class 2
            """),
    dcc.Slider(
        1,
        3,
        step=1,
        id='node2-slider',
        value=1,
    )
])


@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('size-slider', 'value'),
    Input('node1-slider', 'value'),
    Input('node2-slider', 'value'),
)
def update_graph(size, node_number1, node_number2):
    x = []
    y = []
    classes = []
    for number in range(node_number1):
        a, b, c = create_mode(size, "class 1")
        x = np.append(x, a)
        y = np.append(y, b)
        classes = np.append(classes, c)

    for number in range(node_number2):
        a, b, c = create_mode(size, "class 2")
        x = np.append(x, a)
        y = np.append(y, b)
        classes = np.append(classes, c)

    fig = px.scatter(x=x, y=y, color=classes)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 40}, hovermode='closest')
    fig.update_xaxes(title='', type='linear')
    fig.update_yaxes(title='', type='linear')
    return fig


def create_mode(size, name):
    mean = random.uniform(-1, 1)
    standard_deviation = random.uniform(0, 0.1)
    classes = np.full(size, name)

    rng = np.random.default_rng()
    x = rng.normal(mean, standard_deviation, size)
    y = rng.normal(pow(standard_deviation, 2), standard_deviation, size)
    return x, y, classes


if __name__ == '__main__':
    app.run_server(debug=True)
