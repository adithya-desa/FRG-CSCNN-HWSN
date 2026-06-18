import numpy as np
import plotly.graph_objects as go

# IMPORTANT
from sink_path import generate_sink_path

# ---------------------------------------
# LOAD SENSOR DATA
# ---------------------------------------

sensor_data = np.load("data/sensor_data.npy")

positions = sensor_data[:, :2]
energy = sensor_data[:, 2]

# ---------------------------------------
# GENERATE SINK PATH
# ---------------------------------------

sink_path = generate_sink_path(sensor_data)

# ---------------------------------------
# CREATE SMOOTH INTERPOLATED PATH
# ---------------------------------------

smooth_x = []
smooth_y = []

for i in range(len(sink_path) - 1):

    start = sink_path[i]
    end = sink_path[i + 1]

    x_vals = np.linspace(start[0], end[0], 30)
    y_vals = np.linspace(start[1], end[1], 30)

    smooth_x.extend(x_vals)
    smooth_y.extend(y_vals)

smooth_x = np.array(smooth_x)
smooth_y = np.array(smooth_y)

# ---------------------------------------
# SENSOR NODES
# ---------------------------------------

sensor_trace = go.Scatter(
    x=positions[:, 0],
    y=positions[:, 1],
    mode="markers",
    marker=dict(
        size=10,
        color=energy,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(
    title="Energy",
    x=1.08,
    y=0.65,
    len=0.65,
    thickness=20
    )
    ),
    name="Sensor Nodes"
)

# ---------------------------------------
# INITIAL SINK
# ---------------------------------------

sink_trace = go.Scatter(
    x=[smooth_x[0]],
    y=[smooth_y[0]],
    mode="markers",
    marker=dict(
    size=28,
    color="red",
    symbol="star"
    ),
    name="Mobile Sink"
)

# ---------------------------------------
# INITIAL PATH
# ---------------------------------------

path_trace = go.Scatter(
    x=[],
    y=[],
    mode="lines",
    line=dict(
        color="red",
        width=3
    ),
    name="Sink Path"
)

# ---------------------------------------
# ANIMATION FRAMES
# ---------------------------------------

frames = []

for i in range(len(smooth_x)):

    frames.append(

        go.Frame(

            data=[

                sensor_trace,

                go.Scatter(
                    x=smooth_x[:i+1],
                    y=smooth_y[:i+1],
                    mode="lines",
                    line=dict(
                        color="red",
                        width=3
                    ),
                    name="Sink Path"
                ),

                go.Scatter(
                    x=[smooth_x[i]],
                    y=[smooth_y[i]],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="red",
                        symbol="star"
                    ),
                    name="Mobile Sink"
                )

            ],

            name=str(i)

        )

    )

# ---------------------------------------
# FIGURE
# ---------------------------------------

fig = go.Figure(

    data=[
        sensor_trace,
        path_trace,
        sink_trace
    ],

    frames=frames

)

# ---------------------------------------
# LAYOUT
# ---------------------------------------

fig.update_layout(

    title={
        "text": "FRG-CSCNN Mobile Sink Movement",
        "x": 0.5,
        "xanchor": "center"
    },

    xaxis=dict(
        title="X Coordinate",
        range=[0,110]
    ),

    yaxis=dict(
        title="Y Coordinate",
        range=[0,110]
    ),

    autosize=True,

    margin=dict(
    l=50,
    r=250,
    t=80,
    b=50
    ),

    showlegend=False,

    updatemenus=[

        dict(

            type="buttons",

            direction="left",

            x=0.01,
            y=1.15,

            showactive=False,

            buttons=[

                dict(

                    label="▶ Start Animation",

                    method="animate",

                    args=[

                        None,

                        dict(

                            frame=dict(
                                duration=80,
                                redraw=True
                            ),

                            transition=dict(
                                duration=0
                            ),

                            fromcurrent=True

                        )

                    ]

                )

            ]

        )

    ],


    annotations=[

    dict(
        x=1.18,
        y=0.05,

        xref="paper",
        yref="paper",

        text=(
            "<b>Legend</b><br><br>"
            "🟢 Sensor Nodes<br><br>"
            "⭐ Mobile Sink<br><br>"
            "🔴 Sink Path"
        ),

        showarrow=False,

        align="left",

        bordercolor="black",
        borderwidth=1,

        bgcolor="rgba(255,255,255,0.95)",

        font=dict(size=14)
    )

],
)
# ---------------------------------------
# SHOW
# ---------------------------------------

fig.show()