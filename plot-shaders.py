import os
import cupy as cp
import numba
import xarray as xr
import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize
import datashader as ds
from matplotlib.colors import ListedColormap

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


def load_data(path):
    """Load Zarr data using xarray."""
    return xr.open_zarr(path)


@numba.jit
def process_data(sv_values):
    """Process data with Numba."""
    return sv_values[::-1]  # Flip vertically


def create_plot(data, channel):
    """Create an interactive plot using Holoviews and Datashader."""
    sv_data = data.Sv.isel(channel=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']

    # Convert data to CuPy for GPU acceleration
    sv_values = cp.array(sv_data.transpose('ping_time', 'range_sample'))
    sv_values = process_data(sv_values)  # Use Numba for processing

    # Transfer data back to CPU for visualization
    sv_values = cp.asnumpy(sv_values)

    # Create DataArray
    ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])
    print("ds_array:", ds_array)

    # Convert DataArray to HoloViews QuadMesh
    hv_quadmesh = hv.QuadMesh(ds_array, kdims=['ping_time', 'range_sample'], vdims=['Sv'])

    # Use Datashader to rasterize the data for better performance with large datasets
    rasterized_quadmesh = rasterize(hv_quadmesh, aggregator=ds.mean('Sv')).opts(
        opts.QuadMesh(cmap=cmap, colorbar=True, width=1200, height=800,
                      clim=(np.nanmin(sv_values), np.nanmax(sv_values)))
    )

    return rasterized_quadmesh


def print_dataset_info(data):
    print("Dimensions:", data.sizes)
    print("Data variables:", data.data_vars)


def get_cuda_metadata():
    cuda_available = cp.is_available()
    cuda_version = cp.cuda.runtime.getVersion() if cuda_available else "N/A"
    memory_info = cp.cuda.runtime.memGetInfo() if cuda_available else ("N/A", "N/A")

    metadata = {
        "CUDA Available": cuda_available,
        "CUDA Version": cuda_version,
        "Free Memory (Bytes)": memory_info[0],
        "Total Memory (Bytes)": memory_info[1]
    }
    return metadata


def create_cuda_info_panel():
    metadata = get_cuda_metadata()
    text = f"""
    ### CUDA Metadata
    - CUDA Available: {metadata["CUDA Available"]}
    - CUDA Version: {metadata["CUDA Version"]}
    - Free Memory (Bytes): {metadata["Free Memory (Bytes)"]}
    - Total Memory (Bytes): {metadata["Total Memory (Bytes)"]}
    """
    return pn.pane.Markdown(text, sizing_mode='stretch_width')


def create_controls(data):
    channel_selector = pn.widgets.IntSlider(name='Channel', start=0, end=data.sizes['channel'] - 1, step=1, value=0)

    @pn.depends(channel_selector.param.value)
    def update_plot(channel):
        return create_plot(data, channel)

    return pn.Column(channel_selector, update_plot, sizing_mode='stretch_both')


def main():
    zarr_path = 'data/D20070704.zarr'
    data = load_data(zarr_path)

    print_dataset_info(data)
    controls_and_plot = create_controls(data)
    cuda_info_panel = create_cuda_info_panel()

    layout = pn.Column(
        "## Sv Data Visualization",
        cuda_info_panel,
        controls_and_plot,
        sizing_mode='stretch_both'
    )
    layout.servable()


main()
