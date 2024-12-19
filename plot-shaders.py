import os
import xarray as xr
import numpy as np
import panel as pn
import holoviews as hv
from holoviews.operation.datashader import rasterize
import datashader as ds
from bokeh.models import HoverTool, BoxSelectTool, BoxZoomTool, WheelZoomTool, PanTool, ResetTool, SaveTool
import urllib.parse
from colormap import available_colormaps

pn.extension('plotly')
hv.extension('bokeh', width=100)
pn.extension(sizing_mode="stretch_width")


def bytes_to_mb(bytes):
    """Convert bytes to megabytes."""
    return bytes / (1024 ** 2)


def load_data(path):
    """Load Zarr data using xarray."""
    return xr.open_zarr(path)


def print_dataset_info(data):
    print("Dimensions:", data.sizes)
    print("Data variables:", data.data_vars)


# def get_cuda_metadata():
#     cuda_available = cp.is_available()
#     if cuda_available:
#         cuda_version = cp.cuda.runtime.runtimeGetVersion()
#         free_mem, total_mem = cp.cuda.runtime.memGetInfo()
#         free_mem_mb = bytes_to_mb(free_mem)
#         total_mem_mb = bytes_to_mb(total_mem)
#         used_mem_mb = total_mem_mb - free_mem_mb
#     else:
#         cuda_version = "N/A"
#         free_mem_mb, total_mem_mb, used_mem_mb = "N/A", "N/A", "N/A"

#     metadata = {
#         "CUDA Available": cuda_available,
#         "CUDA Version": f"{cuda_version // 1000}.{cuda_version % 1000 // 10}",
#         "Free Memory (MB)": free_mem_mb,
#         "Total Memory (MB)": total_mem_mb,
#         "Used Memory (MB)": used_mem_mb
#     }
#     return metadata


# def create_cuda_info_panel():
#     metadata = get_cuda_metadata()
#     text = f"""
#     ### CUDA Metadata
#     - CUDA Available: {metadata["CUDA Available"]}
#     - CUDA Version: {metadata["CUDA Version"]}
#     - Free Memory (MB): {metadata["Free Memory (MB)"]}
#     - Total Memory (MB): {metadata["Total Memory (MB)"]}
#     - Used Memory (MB): {metadata["Used Memory (MB)"]}
#     """
#     return pn.pane.Markdown(text, sizing_mode='stretch_width')

def create_histogram():
    """Create an empty histogram for displaying selected data."""
    return hv.Histogram(np.histogram([], bins=50), kdims=['Sv'], vdims=['Count']).opts(
        width=400, height=400, xlabel='Sv', ylabel='Count'
    )

def create_controls(data):
    channel_names = [str(chan) for chan in data.channel.values]
    channel_selector = pn.widgets.RadioBoxGroup(name='Channel', options=channel_names)
    colormap_selector = pn.widgets.Select(name='Colormap', options=list(available_colormaps.keys()), value='Viridis')
    histogram = create_histogram()

    @pn.depends(channel_selector.param.value, colormap_selector.param.value)
    def update_plot(channel_name, colormap_name):
        channel = data.channel.values.tolist().index(channel_name)
        colormap = available_colormaps[colormap_name]
        plot = create_plot(data, channel, colormap)
        selection_stream = hv.streams.Selection1D(source=plot)

        def selection_callback(index):
            sv_data = data.Sv.isel(channel=channel)
            sv_data = update_range_sample_with_depth(sv_data).values
            update_histogram(index, sv_data, histogram)

        selection_stream.param.watch(selection_callback, 'index')
        return pn.Row(plot, histogram)


    return pn.Column(channel_selector, colormap_selector, update_plot, sizing_mode='stretch_both')


def get_query_params():
    query_string = pn.state.location.search
    return urllib.parse.parse_qs(query_string)


def update_range_sample_with_depth(sv_data):
    """Update the range_sample coordinate with true depth."""
    first_channel = sv_data["channel"].values[0]
    first_ping_time = sv_data["ping_time"].values[0]

    # Slice the echo_range to get the desired range of values
    selected_echo_range = sv_data["echo_range"].sel(channel=first_channel, ping_time=first_ping_time)
    selected_echo_range = selected_echo_range.values.tolist()
    selected_echo_range = [value + 8.6 for value in selected_echo_range]  # Transducer offset 8.6m

    # Assign the values to the depth coordinate
    sv_data = sv_data.assign_coords(range_sample=selected_echo_range)
    return sv_data

def create_plot(data, channel, colormap):
    """Create an interactive plot using Holoviews and Datashader."""
    sv_data = data.Sv.isel(channel=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']

    sv_values = sv_data.values
    ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])
    hv_quadmesh = hv.QuadMesh(ds_array, kdims=['ping_time', 'range_sample'], vdims=['Sv'])
    rasterized_quadmesh = rasterize(hv_quadmesh, aggregator=ds.mean('Sv'))

    rasterized_quadmesh = rasterized_quadmesh.opts(
        cmap=colormap,
        colorbar=True,
        responsive=True,
        min_height=600,
        clim=(np.nanmin(sv_values), np.nanmax(sv_values)),
        tools=['hover', 'box_select'],
        active_tools=['wheel_zoom'],
        invert_yaxis=True,
        hooks=[lambda plot, element: plot.handles['colorbar'].title('Sv')]
    )

    return rasterized_quadmesh

def update_histogram(selection, sv_data, histogram):
    if selection:
        selected_sv_values = sv_data.flatten()[selection]
        hist, edges = np.histogram(selected_sv_values, bins=50)
        histogram.data = hv.Histogram((edges, hist)).data
    else:
        histogram.data = hv.Histogram(np.histogram([], bins=50)).data

def main():
    params = get_query_params()
    zarr_path = params.get('file', ['data/SE2204_-D20220704-T180334_Sv.zarr'])[0]
    data = load_data(zarr_path)
    data = update_range_sample_with_depth(data)

    controls_and_plot = create_controls(data)
    template = pn.template.FastListTemplate(
        site="OceanStream",
        title='Echogram Viewer',
        sidebar=[controls_and_plot[0], controls_and_plot[1]],
        main=[pn.panel(controls_and_plot[2], sizing_mode="scale_width")],
        accent_base_color="#4099da",
        header_background="#4099da"
    )

    template.servable();


main()
