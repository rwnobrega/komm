import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

app = dash.Dash()

app.config.supress_callback_exceptions = True

app.layout = html.Div(
    [
        html.H1('Komm demo: Walsh--Hadamard sequence'),

        html.P('Documentation reference: http://komm.readthedocs.io/en/latest/komm.WalshHadamardSequence/'),

        html.Div(
            [
                html.Div(
                    id='graphs',
                    style={'width': '80%', 'display': 'inline-block', 'vertical-align': 'top'},
                ),

                html.Div([
                    html.Div([
                        html.P('Length:', style={'margin-bottom': '0px'}),
                        dcc.Slider(
                            id='length-slider',
                            min=1,
                            max=7,
                            value=1,
                            marks={i: str(2**i) for i in range(1, 8)},
                            step=None,
                            updatemode='drag',
                        )],
                        style={'margin-bottom': '36px'},
                    ),

                    html.Div([
                        html.P('Ordering:', style={'margin-bottom': '0px'}),
                        dcc.RadioItems(
                            id='ordering-radio',
                            options=[
                                {'label': 'Natural', 'value': 'natural'},
                                {'label': 'Sequency', 'value': 'sequency'},
                            ],
                            value='natural',
                            labelStyle={'display': 'inline-block'}
                        )],
                        style={'margin-bottom': '24px'},
                    ),

                    html.Div([
                        html.P('Index:', style={'margin-bottom': '0px'}),
                        dcc.Slider(
                            id='index-slider',
                            min=0,
                            value=0,
                            step=1,
                            updatemode='drag',
                        )],
                    )],
                    style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'},
                )
            ],
        )
    ],

    style={'width': '80%', 'margin': 'auto'}
)

# ---

import komm
import numpy as np

@app.callback(
    dash.dependencies.Output(component_id='index-slider', component_property='max'),
    [dash.dependencies.Input(component_id='length-slider', component_property='value')]
)
def _(log_length):
    return 2**log_length - 1

@app.callback(
    dash.dependencies.Output(component_id='index-slider', component_property='marks'),
    [dash.dependencies.Input(component_id='length-slider', component_property='value')]
)
def _(log_length):
    marks = {i: '' for i in range(2**log_length)}
    marks[0] = '0'
    marks[2**log_length - 1] = str(2**log_length - 1)
    return marks

@app.callback(
    dash.dependencies.Output(component_id='graphs', component_property='children'),
    [dash.dependencies.Input(component_id='length-slider', component_property='value'),
     dash.dependencies.Input(component_id='ordering-radio', component_property='value'),
     dash.dependencies.Input(component_id='index-slider', component_property='value')]
)
def barker_sequence_update(log_length, ordering, index):
    length = 2**log_length
    walsh_hadamard = komm.WalshHadamardSequence(length=length, ordering=ordering, index=index)

    figure_sequence = dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=np.arange(length + 1),
                    y=np.pad(walsh_hadamard.polar_sequence, (0, 1), mode='edge'),
                    mode='lines',
                    line=dict(
                        shape='hv',
                    ),
                ),
            ],
            layout=go.Layout(
                title=str(walsh_hadamard),
                xaxis=dict(
                    title='n',
                ),
                yaxis=dict(
                    title='a[n]',
                    dtick=1.0,
                    range=[-1.1, 1.1],
                ),
            ),
        ),
        style={'width': '100%', 'display': 'inline-block'},
        id='sequence',
    )

    return [figure_sequence]


if __name__ == '__main__':
    app.run_server(debug=True)
