import os
import cupy as cp
import xarray as xr
import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize
import datashader as ds
from bokeh.models import HoverTool
from matplotlib.colors import ListedColormap
from bokeh.palettes import Viridis256

pn.extension('plotly')
hv.extension('bokeh', width=100)

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
    """Create an interactive plot using Holoviews and Datashader."""
    sv_data = data.Sv.isel(channel=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']

    # Convert data to CuPy for GPU acceleration
    sv_values = cp.array(sv_data.transpose('ping_time', 'range_sample'))
    sv_values = process_data_with_cupy(sv_values)  # Use CuPy for processing

    # Transfer data back to CPU for visualization
    sv_values = cp.asnumpy(sv_values)

    # Create DataArray
    ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])

    # Convert DataArray to HoloViews QuadMesh
    hv_quadmesh = hv.QuadMesh(ds_array, kdims=['ping_time', 'range_sample'], vdims=['Sv']).opts(
        cmap='viridis',
        colorbar=True,
        width=1200,
        height=800,
        clim=(np.nanmin(sv_values), np.nanmax(sv_values)),
        tools=['hover'],
        invert_yaxis=True,  # Flip vertically
        hooks=[lambda plot, element: plot.handles['colorbar'].set_label('Sv')]
    )

    rasterized_quadmesh = rasterize(hv_quadmesh, aggregator=ds.mean('Sv')).opts(
        opts.QuadMesh(
            tools=['hover'],
            hover_line_color='white',
            hover_fill_color='blue',
            active_tools=['wheel_zoom']
        )
    )

    return rasterized_quadmesh

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
        used_mem_mb = total_mem_mb - free_mem_mb
    else:
        cuda_version = "N/A"
        free_mem_mb, total_mem_mb, used_mem_mb = "N/A", "N/A", "N/A"

    metadata = {
        "CUDA Available": cuda_available,
        "CUDA Version": f"{cuda_version // 1000}.{cuda_version % 1000 // 10}",
        "Free Memory (MB)": free_mem_mb,
        "Total Memory (MB)": total_mem_mb,
        "Used Memory (MB)": used_mem_mb
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
    - Used Memory (MB): {metadata["Used Memory (MB)"]}
    """
    return pn.pane.Markdown(text, sizing_mode='stretch_width')

def create_controls(data):
    channel_selector = pn.widgets.IntSlider(name='Channel', start=0, end=data.sizes['channel'] - 1, step=1, value=0)

    @pn.depends(channel_selector.param.value)
    def update_plot(channel):
        return create_plot(data, channel)

    return pn.Column(channel_selector, update_plot, sizing_mode='stretch_both')

def update_metadata(event):
    """Update CUDA metadata panel on echogram plot events."""
    cuda_info_panel.object = create_cuda_info_panel()

def main():
    zarr_path = 'data/D20070704.zarr'
    data = load_data(zarr_path)

    print_dataset_info(data)
    controls_and_plot = create_controls(data)
    global cuda_info_panel
    cuda_info_panel = create_cuda_info_panel()

    # Adjust layout to make echogram visualization occupy at least 90% of the screen
    sidebar = pn.Column(
        cuda_info_panel,
        controls_and_plot[0],  # Only include the channel selector in the sidebar
        width=300,
        sizing_mode='stretch_height'
    )

    main_content = pn.Column(
        controls_and_plot[1],  # Include the echogram plot
        sizing_mode='stretch_both'
    )

    # Add event listeners to update CUDA metadata on echogram interactions
    hv_quadmesh = controls_and_plot[1][0]
    hv_quadmesh.param.watch(update_metadata, ['value'])

    layout = pn.Row(sidebar, main_content, sizing_mode='stretch_both')
    layout.servable()

main()
