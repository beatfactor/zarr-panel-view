import xarray as xr
import numpy as np
import panel as pn
import plotly.express as px
import holoviews as hv
import re
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs

pn.extension('plotly')
hv.extension('bokeh')


def load_data(path):
    """Load Zarr data using xarray."""
    return xr.open_zarr(path)


def create_plot(data, channel):
    """Create an interactive plot using Plotly."""
    sv_data = data.Sv.sel(channel=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']
    sv_values = sv_data.transpose('ping_time', 'range_sample')  # Ensure dimensions match

    # Create DataArray
    ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])
    # Convert the DataArray to a numpy array for Plotly
    img_array = np.flipud(ds_array.values.T)
    fig = px.imshow(
        img_array,
        labels={'color': 'Sv (dB)', 'x': 'Time', 'y': 'Depth (m)'},
        origin='lower',
        color_continuous_scale="viridis",
        aspect='auto'
    )

    # Update the layout with more specific axis labels and add a title
    fig.update_layout(
        width=1200,
        height=900,
        autosize=True,
        title='Fisheries Acoustics Echogram',
        xaxis_title='Time',
        yaxis_title='Depth (m)',
    )

    # Update the color scale limits
    fig.update_coloraxes(cmin=-75, cmax=-35)

    # Add tooltips with more detailed information
    fig.update_traces(
        hovertemplate='Time: %{x}<br>Depth: %{y} m<br>Sv: %{z:.2f} dB'
    )

    return fig


# def create_plot(data, channel):
#     """Create an interactive plot using Plotly."""
#     sv_data = data.Sv.sel(channel=channel)
#     ping_time = sv_data['ping_time']
#     range_sample = sv_data['range_sample']
#     sv_values = sv_data.transpose('ping_time', 'range_sample')  # Ensure dimensions match
#
#     # Create DataArray
#     ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])
#     # Convert the DataArray to a numpy array for Plotly
#     img_array = np.flipud(ds_array.values.T)
#     fig = px.imshow(
#         img_array,
#         labels={'color': 'Sv (dB)'},
#         origin='lower',
#         color_continuous_scale="viridis"
#     )
#     fig.update_layout(width=1400, height=900, autosize=False)
#     fig.update_coloraxes(cmin=-75, cmax=-35)
#     return fig


def print_dataset_info(data):
    print("Dimensions:", data.sizes)
    print("Data variables:", data.data_vars)


def parse_image_url(url):
    match = re.match(r'(.+_-D\d{8}-T\d{6})_.*\.png', url)
    if match:
        return match.group(1)
    return url


def create_controls(data):
    channel_options = [(str(channel)) for channel in data.channel.values]
    channel_selector = pn.widgets.Select(name='Channel', options=channel_options)

    @pn.depends(channel_selector)
    def update_plot(channel):
        return pn.pane.Plotly(create_plot(data, channel), sizing_mode='stretch_both')

    return pn.Column(channel_selector, update_plot)


def get_zarr_file_from_query():
    query = pn.state.location.search
    params = parse_qs(query[1:])  # Skip the initial '?'
    zarr_file = params.get('zarr', [''])[0]
    return zarr_file


def read_and_sort_images(path):
    """Read all images from the given path and sort them by date and time."""
    image_files = [f for f in os.listdir(path) if f.endswith('.png')]
    parsed_images = {}

    for image_file in image_files:
        match = re.match(r'(.+_-D\d{8}-T\d{6})_(.*)\.png', image_file)
        if match:
            base_name = match.group(1)
            channel_name = match.group(2)
            timestamp_str = base_name.split('-D')[1].split('-T')
            date_str = timestamp_str[0]
            time_str = timestamp_str[1]
            timestamp = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

            if base_name not in parsed_images:
                parsed_images[base_name] = {
                    'timestamp': timestamp,
                    'channels': {}
                }
            if channel_name not in parsed_images[base_name]['channels']:
                parsed_images[base_name]['channels'][channel_name] = []
            parsed_images[base_name]['channels'][channel_name].append(image_file)

    # Sort by timestamp
    sorted_images = sorted(parsed_images.items(), key=lambda x: x[1]['timestamp'])

    # Return sorted list of key-value items
    return [{key: value['channels']} for key, value in sorted_images]


def create_carousel(zarr_file, image_path, selected_channel):
    image_info = read_and_sort_images(image_path)
    current_file = f"{zarr_file}.png"

    image_panes = []
    for item in image_info:
        for key, channels in item.items():
            if selected_channel in channels:
                first_image = channels[selected_channel][0]
                image_panes.append(
                    pn.pane.HTML(
                        f'<div class="image-container">'
                        f'<a href="/?zarr={key}" class="image-link" title="{key}">'
                        f'<img src="http://192.168.0.39:3000/echograms/{first_image}" loading="lazy" class="image {"active-image" if current_file == first_image else ""}" /></a></div>',
                        sizing_mode='fixed'
                    )
                )

    toggle = pn.widgets.Toggle(name="▼", button_type="primary")
    image_row = pn.Row(*image_panes, sizing_mode='stretch_width', css_classes=['carousel'])

    carousel = pn.Column(
        toggle,
        pn.layout.HSpacer(height=10),
        image_row,
        sizing_mode='stretch_both'
    )

    @pn.depends(toggle.param.value)
    def toggle_carousel(show):
        toggle.name = "▲" if show else "▼"
        return image_row if show else pn.Spacer(height=0)

    return pn.Column(toggle, toggle_carousel)


# def create_carousel(zarr_file):
#     image_urls = [
#         'SE2204_-D20220704-T162237_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T171306_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T180334_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T185406_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T162237_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T171306_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T180334_GPT_00907205c5fa-1_ES70-7C_ES.png',
#         'SE2204_-D20220704-T185406_GPT_00907205c5fa-1_ES70-7C_ES.png'
#     ]
#     image_panes = [pn.pane.HTML(f'<div class="image-container">'
#                                 f'<a href="/?zarr_file={parse_image_url(url)}" class="image-link">'
#                                 f'<img src="http://192.168.0.39:3000/echograms/{url}" loading="lazy" class="image" /></a></div>', sizing_mode='fixed')
#                    for url in image_urls]
#
#     toggle = pn.widgets.Toggle(name="▼", button_type="primary")
#     image_row = pn.Row(*image_panes, sizing_mode='stretch_width', css_classes=['carousel'])
#
#     carousel = pn.Column(
#         toggle,
#         pn.layout.HSpacer(height=10),
#         image_row,
#         sizing_mode='stretch_both'
#     )
#
#     @pn.depends(toggle.param.value)
#     def toggle_carousel(show):
#         toggle.name = "▲" if show else "▼"
#         return image_row if show else pn.Spacer(height=0)
#
#     return pn.Column(toggle, toggle_carousel)


def main():
    zarr_file = get_zarr_file_from_query()
    if not zarr_file:
        zarr_file = 'data/D20070704.zarr'  # Default file for demonstration

    data = load_data(zarr_file)
    controls_and_plot = create_controls(data)

    @pn.depends(controls_and_plot[0])
    def update_carousel(channel):
        print("Channel:", channel)
        return create_carousel(zarr_file, 'data/echograms', channel)

    print_dataset_info(data)

    layout = pn.Column(
        "## Fisheries Acoustic Sv Data Visualization",
        controls_and_plot,
        update_carousel,
        sizing_mode='stretch_both'
    )
    layout.servable()


# CSS for carousel
carousel_css = """
.carousel {
    display: flex;
    overflow-x: auto;
    white-space: nowrap;
}
.image-container {
    display: block;
    width: 440px;
    height: 240px;
    position: relative;
    overflow: hidden;
    border: 2px solid #000;
}
.image-link {
    display: block;
    position: absolute;
    top: -6px;
    left: -20px;
    right: 0;
    height: 234px;
}
.image-container:hover {
    border: 2px solid red;
}

.image {
    height: 350px;
}

.bk-panel-models-markup-HTML {
    margin: 0;
    padding: 0;
}
"""

# Inject CSS
pn.extension(raw_css=[carousel_css])

main()
