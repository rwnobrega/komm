import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

app = dash.Dash()

app.config.supress_callback_exceptions = True

app.layout = html.Div([
    html.H1('Komm demo: Barker sequence'),

    html.P('Documentation reference: http://komm.readthedocs.io/en/latest/komm.BarkerSequence/'),

    html.Label('Length:'),

    html.Div([
        dcc.Slider(
            id='barker-length-slider',
            min=2,
            max=13,
            value=2,
            marks={length: str(length) for length in [2, 3, 4, 5, 7, 11, 13]},
            step=None,
            updatemode='drag',
        )
    ], style={'margin-bottom': '25px', 'align': 'center'}),

    html.Div(
        id='barker-graphs',
    ),
], style={'width': '80%', 'margin': 'auto'})

# ---

import komm
import numpy as np


@app.callback(
    dash.dependencies.Output(component_id='barker-graphs', component_property='children'),
    [dash.dependencies.Input(component_id='barker-length-slider', component_property='value')]
)
def barker_sequence_update(length):
    barker = komm.BarkerSequence(length=length)
    shifts = np.arange(-length - 1, length + 2)

    figure_barker_sequence = dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=np.arange(length + 1),
                    y=np.pad(barker.polar_sequence, (0, 1), mode='edge'),
                    mode='lines',
                    line=dict(
                        shape='hv',
                    ),
                ),
            ],
            layout=go.Layout(
                title=str(barker),
                xaxis=dict(
                    title='n',
                    dtick=1.0,
                ),
                yaxis=dict(
                    title='a[n]',
                    dtick=1.0,
                ),
            ),
        ),
        style={'display': 'inline-block', 'width': '50%'},
        id='barker-sequence',
    )

    figure_barker_autocorrelation = dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=shifts,
                    y=barker.autocorrelation(shifts),
                    mode='lines',
                ),
            ],
            layout=go.Layout(
                title='Autocorrelation',
                xaxis=dict(
                    title='ℓ',
                ),
                yaxis=dict(
                    title='R[ℓ]',
                ),
            ),
        ),
        style={'display': 'inline-block', 'width': '50%'},
        id='barker-autocorrelation',
    )

    return [figure_barker_sequence, figure_barker_autocorrelation]


if __name__ == '__main__':
    app.run_server(debug=True)
