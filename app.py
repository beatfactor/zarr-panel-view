import os
import cupy as cp
import xarray as xr
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.colors import ListedColormap

# Define the custom colormap
__cmap_colors = {
    'ek500': {
        'rgb': (
                np.array(
                    [
                        [159, 159, 159],  # light grey
                        [95, 95, 95],  # grey
                        [0, 0, 255],  # dark blue
                        [0, 0, 127],  # blue
                        [0, 191, 0],  # green
                        [0, 127, 0],  # dark green
                        [255, 255, 0],  # yellow
                        [255, 127, 0],  # orange
                        [255, 0, 191],  # pink
                        [255, 0, 0],  # red
                        [166, 83, 60],  # light brown
                    ]
                )
                / 255
        ),
        'under': '1',  # white
        'over': np.array([120, 60, 40]) / 255,  # dark brown
    }
}

def _create_cmap(rgb, under=None, over=None):
    cmap = ListedColormap(rgb)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)
    return cmap

# Create and register the colormap
colors_d = __cmap_colors['ek500']
rgb = colors_d['rgb']
cmap = _create_cmap(rgb, under=colors_d.get('under', None), over=colors_d.get('over', None))

def bytes_to_mb(bytes):
    """Convert bytes to megabytes."""
    return bytes / (1024 ** 2)

def load_data(path):
    """Load Zarr data using xarray."""
    return xr.open_zarr(path)

def process_data_with_cupy(sv_values):
    """Process data using CuPy."""
    return cp.flipud(sv_values)  # Flip vertically

def create_plot(data, channel):
    """Create an interactive plot using Plotly."""
    sv_data = data.Sv.isel(channel=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']

    # Convert data to CuPy for GPU acceleration
    sv_values = cp.array(sv_data.transpose('ping_time', 'range_sample'))
    sv_values = process_data_with_cupy(sv_values)  # Use CuPy for processing

    # Transfer data back to CPU for visualization
    sv_values = cp.asnumpy(sv_values)

    fig = go.Figure(
        data=go.Heatmap(
            z=sv_values,
            x=ping_time,
            y=range_sample,
            colorscale=[(i / (len(rgb) - 1), 'rgb({},{},{})'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))) for i, color in enumerate(rgb)]
        )
    )
    fig.update_layout(width=1200, height=800)

    return fig

def print_dataset_info(data):
    print("Dimensions:", data.sizes)
    print("Data variables:", data.data_vars)

def get_cuda_metadata():
    cuda_available = cp.is_available()
    if cuda_available:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        free_mem_mb = bytes_to_mb(free_mem)
        total_mem_mb = bytes_to_mb(total_mem)
    else:
        cuda_version = "N/A"
        free_mem_mb, total_mem_mb = "N/A", "N/A"

    metadata = {
        "CUDA Available": cuda_available,
        "CUDA Version": f"{cuda_version // 1000}.{cuda_version % 1000 // 10}",
        "Free Memory (MB)": free_mem_mb,
        "Total Memory (MB)": total_mem_mb
    }
    return metadata

def create_cuda_info_panel():
    metadata = get_cuda_metadata()
    text = f"""
    ### CUDA Metadata
    - CUDA Available: {metadata["CUDA Available"]}
    - CUDA Version: {metadata["CUDA Version"]}
    - Free Memory (MB): {metadata["Free Memory (MB)"]}
    - Total Memory (MB): {metadata["Total Memory (MB)"]}
    """
    return text

# Load data
zarr_path = 'data/D20070704.zarr'
data = load_data(zarr_path)
print_dataset_info(data)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H2('CUDA Metadata'),
        dcc.Markdown(create_cuda_info_panel()),
        html.H2('Channel Selection'),
        dcc.Slider(
            id='channel-slider',
            min=0,
            max=data.sizes['channel'] - 1,
            step=1,
            value=0,
            marks={i: str(i) for i in range(data.sizes['channel'])}
        )
    ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
    html.Div([
        dcc.Graph(id='echogram')
    ], style={'width': '80%', 'display': 'inline-block', 'padding': '10px'})
])

@app.callback(
    Output('echogram', 'figure'),
    Input('channel-slider', 'value')
)
def update_echogram(channel):
    return create_plot(data, channel)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
