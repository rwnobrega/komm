import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

app = dash.Dash()

app.config.supress_callback_exceptions = True

app.layout = html.Div([
    html.H1('Komm demo: Linear-feedback shift register (LFSR) sequence -- Maximum-length sequence (MLS)'),

    html.P('Documentation reference: http://komm.readthedocs.io/en/latest/komm.LFSRSequence/'),

    html.Label('Degree:'),

    html.Div([
        dcc.Slider(
            id='degree-slider',
            min=2,
            max=7,
            value=2,
            marks={length: str(length) for length in range(2, 8)},
            step=None,
            updatemode='drag',
        )
    ], style={'margin-bottom': '25px', 'align': 'center'}),

    html.Div(
        id='graphs',
    ),
], style={'width': '80%', 'margin': 'auto'})

# ---

import komm
import numpy as np


@app.callback(
    dash.dependencies.Output(component_id='graphs', component_property='children'),
    [dash.dependencies.Input(component_id='degree-slider', component_property='value')]
)
def lfsr_sequence_update(degree):
    lfsr = komm.LFSRSequence.maximum_length_sequence(degree=degree)
    length = lfsr.length
    shifts = np.arange(-2*length + 1, 2*length)

    figure_sequence = dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=np.arange(length + 1),
                    y=np.pad(lfsr.polar_sequence, (0, 1), mode='edge'),
                    mode='lines',
                    line=dict(
                        shape='hv',
                    ),
                ),
            ],
            layout=go.Layout(
                title=str(lfsr),
                yaxis=dict(
                    title='a[n]',
                    dtick=1.0,
                ),
            ),
        ),
        style={'display': 'inline-block', 'width': '50%'},
        id='sequence',
    )

    figure_cyclic_autocorrelation = dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=shifts,
                    y=lfsr.cyclic_autocorrelation(shifts, normalized=True),
                    mode='lines',
                ),
            ],
            layout=go.Layout(
                title='Cyclic autocorrelation (normalized)',
                xaxis=dict(
                    title='ℓ',
                ),
                yaxis=dict(
                    title='R~[ℓ]',
                ),
            ),
        ),
        style={'display': 'inline-block', 'width': '50%'},
        id='cyclic-autocorrelation',
    )

    return [figure_sequence, figure_cyclic_autocorrelation]


if __name__ == '__main__':
    app.run_server(debug=True)
