import os

os.environ['NUMBA_THREADING_LAYER'] = 'tbb'

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


def create_plot(data, channel):
    """Create an interactive plot using Holoviews and Datashader."""
    sv_data = data.Sv.isel(channel=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']
    sv_values = sv_data.transpose('ping_time', 'range_sample')[::-1]  # Ensure dimensions match and flip vertically

    # Create DataArray
    ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])
    print("ds_array:", ds_array)

    # Convert DataArray to HoloViews QuadMesh
    hv_quadmesh = hv.QuadMesh(ds_array, kdims=['ping_time', 'range_sample'], vdims=['Sv'])

    # Use Datashader to rasterize the data for better performance with large datasets
    rasterized_quadmesh = rasterize(hv_quadmesh, aggregator=ds.mean('Sv')).opts(
        opts.QuadMesh(cmap=cmap, colorbar=True, width=1200, height=800, clim=(np.nanmin(sv_values), np.nanmax(sv_values)))
    )

    return rasterized_quadmesh


def print_dataset_info(data):
    print("Dimensions:", data.sizes)
    print("Data variables:", data.data_vars)


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

    layout = pn.Column("## Fisheries Acoustic Sv Data Visualization", controls_and_plot, sizing_mode='stretch_both')
    layout.servable()


main()
