"""ACLED Data Science Laboratory

This is a bokeh server application. It can be launched with

    bokeh serve --log-level debug /path/to/server/

and connect with a browser to

    http://localhost:5006/server
"""

from datetime import datetime
# Add parent folder modules/ to sys.path
import sys
sys.path.insert(0, 'modules')

from bokeh.events import MouseMove, Tap
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, LogColorMapper, Range1d, Plot, Text
from bokeh.models.glyphs import Rect
from bokeh.models.widgets import Div, CheckboxGroup, TextInput, Select, Button, Slider
from bokeh.palettes import OrRd9 as palette; palette.reverse()
from bokeh.plotting import curdoc, figure
from bokeh.settings import logging as log

import numpy as np

from presets import presets as selected_presets
from presets import names as selected_presets_names
from ProphetScripts import run_prophet_prediction
from ImportShapefile import ImportShapefile
from acled_preprocess import acled_preprocess


app_name = "ACLED Data Science Laboratory"
log.debug("STARTED SESSION " + app_name)

# Set app title
curdoc().title = app_name

# Set dimension of Africa map and color mapper function
africa_map_plot_width = 850
color_mapper_type = 'log'

# Predict events or fatailities (experimental Prohpet value)
variable = 'events'
if variable == 'fatalities':
    pivot_aggr_fn = 'sum'
else:
    pivot_aggr_fn = 'count'

# Get ACLED pandas.DataFrame from server context
df_full = curdoc().df_full

# Preprocess pandas and geopandas data frames
link_ESRI_shp = 'data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
gpd_df = ImportShapefile(link_ESRI_shp).get_df()
df_full, gpd_df = acled_preprocess(df_full, gpd_df)
df_piv = df_full.pivot_table(index='event_date',
                             columns='country',
                             values='fatalities',
                             aggfunc=pivot_aggr_fn)

# Aggregate over time window
time_window = '1M'
df_piv = df_piv.resample(time_window,
                         closed='left',
                         label='right'
                        ).sum().fillna(value=0).T

gpd_df['value'] = df_piv.iloc[:, 0]

# INCOMPATIBILITY FIX
# In the ESRI shapefile, Angola is a 'MultiPolygon' object. This means that
# two separate polygons make up Angola (the smaller being the Cabinda province).
# In previous versions of Bokeh, this could be handled by inserting an 'np.nan'
# in between the two lists of coordinates. However, this leads to a Bokeh
# ValueError stating the type is not JSON serializable. We remove Cabinda below.
gpd_df.set_value('Angola', 'x', gpd_df.loc['Angola', 'x'][:66])
gpd_df.set_value('Angola', 'y', gpd_df.loc['Angola', 'y'][:66])

# Define data sources
africa_map_cds = ColumnDataSource(gpd_df.sort_index())
colormap_legend_cds = ColumnDataSource(dict(tick_max=['{:.0f}'.format(
    max(africa_map_cds.data['value']))]))
prophet_cds_y_hat_fit = ColumnDataSource(data=dict(x=[], y=[]))
prophet_cds_y_train = ColumnDataSource(data=dict(x=[], y=[]))
prophet_cds_y_test = ColumnDataSource(data=dict(x=[], y=[]))
prophet_cds_bband_uncertainty = ColumnDataSource(dict(x=[], y=[]))

# Define selection box parameters and data source
'''
  Order of selection box coordinates:
  (0) ----------- (1)    (0) = (x0,y1)
     |           |       (1) = (x1,y1)
     |           |       (2) = (x1,y0)
     |           |       (3) = (x0,y0)
  (3) ----------- (2)
'''
x_min, x_max = -18, 52
y_min, y_max = -36, 38
x_range, y_range = [16, 35], [-15, 10]

map_select_area_cds = ColumnDataSource(
    dict(x=[[x_range[0], x_range[1], x_range[1], x_range[0]]],
         y=[[y_range[1], y_range[1], y_range[0], y_range[0]]],
         name=['map_select']))


def africa_map_add_legend(palette, africa_map_plot_width):
    """This creates the colormap legend under the map of Africa.

    Input:
        palette: Defines colormap to use in legend
        africa_map_plot_width: Width of map plot used for scaling/positioning

    Credit:
    Adds glyphs for legend. Based on [1]

    References:
        [1] https://github.com/chdoig/scipy2015-blaze-bokeh/blob/master
            /1.5%20Glyphs%20-%20Legend%20(solution).ipynb
    """
    # Set ranges
    xdr = Range1d(0, africa_map_plot_width)
    ydr = Range1d(0, 60)

    # Create plot
    legend = Plot(
        x_range=xdr,
        y_range=ydr,
        plot_width=africa_map_plot_width,
        plot_height=60,
        min_border=0,
        toolbar_location=None,
        outline_line_color="#FFFFFF",
    )

    # Add the legend color boxes and text
    width = 40
    height = 20
    for i, color in enumerate(palette):
        rect = Rect(
            x=(width * (i + 1)), y=40,
            width=width*1, height=height,
            fill_color=color, line_color='black'
        )
        legend.add_glyph(rect)

    tick_min = Text(x=width/2, y=0, text=['0'])
    legend.add_glyph(tick_min)
    text_scale = Text(x=width*3, y=0, text=['Logarithmic scale'])
    legend.add_glyph(text_scale)
    tick_max = Text(x=width*(len(palette)), y=0, text='tick_max')

    legend.add_glyph(colormap_legend_cds, tick_max)

    return legend

# Area selection box callbacks
area_select_active = False
def update_box_coords():
    """Push new selection box coordinates to update map plot."""
    map_select_area_cds.data['x'] = [[x_range[0], x_range[1], x_range[1], x_range[0]]]
    map_select_area_cds.data['y'] = [[y_range[1], y_range[1], y_range[0], y_range[0]]]

def select_box_event_tap(event):
    """Callback to handle mouse click on map."""
    global area_select_active
    area_select_active = not area_select_active
    if area_select_active:
        x_range[0], y_range[0] = event.x, event.y
        x_range[1], y_range[1] = event.x, event.y
        update_box_coords()
    else:
        x_range[0], x_range[1] = min(x_range), max(x_range)
        y_range[0], y_range[1] = min(y_range), max(y_range)

def select_box_event_mousemove(event):
    """Callback to handle mouse movement on map if area selection is active."""
    if area_select_active:
        x_range[1] = min(x_max, max(x_min, event.x))
        y_range[1] = min(y_max, max(y_min, event.y))
        update_box_coords()
        text_status.text = get_selected_area_coordinates_text()


def africa_map_figure(africa_map_cds, map_select_area_cds, data='value',
                      title='', text_font_size=None,
                      africa_map_plot_width=700):
    """Create plot of Africa map with contours, tools and callbacks.

    Input:
        africa_map_cds: Africa map data source including columns
            'x' and 'y' that cointain exterior coordinates of the contries contours.
            One column with values used for colouring (see 'data')
            'name': Country name
        map_select_area_cds:  Area selection box data source
        data: String containing name of column in 'df' to fill contours
        title: Title text of figure
        text_font_size: Title text font size
    Returns:
        bokeh figure
    """
    if color_mapper_type == 'log':
        color_mapper = LogColorMapper(palette=palette)
    elif color_mapper_type == 'linear':
        color_mapper = LinearColorMapper(palette=palette)

    # Create Africa figure
    fig = figure(title=title, tools='pan,wheel_zoom,reset,save',
                 plot_width=africa_map_plot_width, plot_height=africa_map_plot_width,
                 active_drag=None)
    fig.xaxis[0].axis_label = 'Longitude'
    fig.yaxis[0].axis_label = 'Latitude'
    if text_font_size is not None:
        fig.title.text_font_size = text_font_size

    # Draw Africa
    contour = fig.patches('x', 'y', source=africa_map_cds,
                          fill_color={'field': data, 'transform': color_mapper},
                          fill_alpha=1, line_color="black", line_width=0.3)

    # Add area selection box and callbacks
    fig.patches('x', 'y', source=map_select_area_cds, color=["navy"],
                alpha=[0.3], line_width=2)
    fig.on_event(Tap, select_box_event_tap)
    fig.on_event(MouseMove, select_box_event_mousemove)

    # Add hover tool to display information when mouse is on Africa map
    hover = HoverTool(renderers=[contour])
    hover.tooltips = [("Country", "@name"), (data.capitalize(), "@"+data)]
    fig.add_tools(hover)

    return fig


def update_prophet_datasource(forecast, df_train, df_test):
    """Push prophet results onto result plot.

    Inputs:
        forecast: Data from trained prophet model
        df_train: Raw data used in training
        df_test: Raw data for validation (same preprocessing as for df_train)
    """

    # Define Bollinger Bands.
    upperband = forecast['yhat_lower']
    lowerband = forecast['yhat_upper']
    x_data = forecast['ds']

    # Update data sources
    prophet_cds_y_hat_fit.data = dict(x=forecast['ds'], y=forecast['yhat'])
    prophet_cds_y_train.data = dict(x=df_train['ds'], y=df_train['y'])
    prophet_cds_y_test.data = dict(x=df_test['ds'], y=df_test['y'])
    prophet_cds_bband_uncertainty.data['x'] = np.append(x_data, x_data[::-1])
    prophet_cds_bband_uncertainty.data['y'] = np.append(lowerband, upperband[::-1])


def get_newest_date_in_db_text():
    """Returns information string of the newest event in the database."""
    return "<b>Newest event in database is {}</b>".format(
        df_full['event_date'].max().strftime("%Y-%m-%d"))

def get_selected_area_coordinates_text():
    """Returns information string of the coordinates for the current selected area."""
    return "Selected area: ({:.1f},{:.1f}), ({:.1f},{:.1f})".format(
        x_range[0], y_range[0], x_range[1], y_range[1])

def get_start_index():
    """Return usable array start index calculated from select time window."""
    return max(0, slider_et.value - slider_ws.value - 1)

def get_end_index():
    """Return usable array end index calculated from select time window."""
    return min(number_of_months, slider_et.value - 1)

def get_period_text():
    """Returns information string to display Start Month and End Month."""
    return "<table><tr><td>Start month:</td><td>{}</td></tr>" \
           "<tr><td>End month:</td><td>{}</td></tr></table>".format(
               df_piv.columns[get_start_index()].strftime("%Y-%m"),
               df_piv.columns[get_end_index()].strftime("%Y-%m"))

def update_time_window_datasources():
    """Push new data when start/end time sliders change."""
    text_period.text = get_period_text()
    africa_map_cds.data['value'] = df_piv.iloc[:, get_start_index():get_end_index()+1].sum(axis=1)
    colormap_legend_cds.data['tick_max'] = ['{:.0f}'.format(max(africa_map_cds.data['value']))]

def slider_et_callback(attrname, old, new):
    """Callback for Last month slider."""
    slider_ws.update(end=slider_et.value-1)
    slider_ws.value = min(slider_ws.value, slider_et.value-1)
    update_time_window_datasources()

def slider_ws_callback(attrname, old, new):
    """Callback for Months to accumulate slider."""
    update_time_window_datasources()


# Text: Area and time
text_dataset = Div(
    text='<h3>Area and time parameters</h3><p><b>Select area on map:</b> Tap, move, tap</p>')


# Drop-down list: Example presets
select_preset = Select(title='Presets', value='',
                       options=[''] + selected_presets_names)
def select_preset_callback(attr, old, new):
    """Callback to populate controls on preset selection."""
    try:
        i = selected_presets_names.index(new)
    except ValueError:
        # Do nothing on selection of non-existing preset
        return
    slider_et.value = selected_presets[i]['current_month']
    slider_ws.value = selected_presets[i]['acc_months']
    x_range[0] = selected_presets[i]['selected_area'][0]    # X-Left
    x_range[1] = selected_presets[i]['selected_area'][1]    # X-Right
    y_range[0] = selected_presets[i]['selected_area'][2]    # Y-lower
    y_range[1] = selected_presets[i]['selected_area'][3]    # Y-upper
    update_box_coords()
    checkbox_log_scale.active = [0] if selected_presets[i]['log_of_y'] else []
    slider_window_size.value = selected_presets[i]['time_window']
    text_reg_changepoint.value = str(selected_presets[i]['reg_changepoint'])
    text_reg_season.value = str(selected_presets[i]['reg_season'])
    slider_freq_days.value = selected_presets[i]['freq_days']
    text_periods.value = str(selected_presets[i]['periods'])
select_preset.on_change('value', select_preset_callback)


# Slider: Last month in time window
number_of_months = df_piv.shape[1]
slider_et = Slider(start=2, end=number_of_months, value=number_of_months, step=1,  # current month
                   title="Last month in time window")
slider_et.on_change('value', slider_et_callback)


# Slider: Previous months to accumulate
slider_ws = Slider(start=1, end=slider_et.value, value=36, step=1,   # acc months
                   title="Previous months to accumulate")
slider_ws.on_change('value', slider_ws_callback)


# Text: Start month / End month
text_period = Div(text=get_period_text())


# Create widgetbox of parameters
widgets_dataset = widgetbox(select_preset, text_dataset, slider_et, slider_ws, text_period)


# Vairous widgets: Prophet parameter controls
text_prophet_parameters = Div(text='<h3>Prophet parameters</h3>')
checkbox_log_scale = CheckboxGroup(labels=['Logarithmic scale'], active=[])
slider_window_size = Slider(start=1, end=21, value=1, step=1,
                            title='Sliding window size')
text_reg_changepoint = TextInput(value='1.0', title='Regularizer changepoint (decimal number)')
text_reg_season = TextInput(value='1.0', title='Regularizer season (decimal number)')
slider_freq_days = Slider(start=1, end=21, value=1, step=1,
                          title='Frequency of prediction (days in periods)')
text_periods = TextInput(value='1', title='Number of periods to predict (integer)')
# Button: Run Prophet!
button_run_prophet = Button(label="Run Prohpet!")
def button_run_prophet_callback():
    """Callback to prepare and run Prohpet."""

    log.debug("***PROPHET*PREDICT*** (x_range={},yrange={}, start={},end={})".format(
        x_range, y_range,
        df_piv.columns[get_start_index()].strftime("%Y-%m"),
        df_piv.columns[get_end_index()].strftime("%Y-%m")))

    # Ensure Regularizer values are decimal numbers
    try:
        reg_changepoint = float(text_reg_changepoint.value)
        reg_season = float(text_reg_season.value)
    except ValueError:
        text_status.text = 'Regularizer: Not a decimal number!'
        return
    # Ensure Number of periods value is an integer
    try:
        periods = int(text_periods.value)
    except ValueError:
        text_status.text = 'Number of periods: Not a valid integer!'
        return

    # Prepare Prophet preprocessor parameters
    prophet_preprocessing_params = {
        'df': df_full,
        'x_range' : x_range,
        'y_range' : y_range,
        'date_range' : [df_piv.columns[get_start_index()], df_piv.columns[get_end_index()]],
        'log_of_y': 0 in checkbox_log_scale.active,
        'time_window': slider_window_size.value,
        'pivot_aggr_fn': pivot_aggr_fn}

    text_status.text = 'Running Prophet ...'

    # Try running Prophet and catch exceptions
    try:
        start_time = datetime.now()
        pred = run_prophet_prediction(
            prophet_preprocessing_params, reg_changepoint, reg_season,
            periods=periods, freq=slider_freq_days.value)
        update_prophet_datasource(*pred)
        end_time = datetime.now()
        text_status.text = 'Prophet completed in {:d} seconds'.format(
            (end_time - start_time).seconds)
    except KeyError as ex:
        text_status.text = 'Prophet error: No events within selection box {}'.format(str(ex))
    except Exception as ex:
        text_status.text = 'Prophet error: {} (likely insufficient data)'.format(str(ex))
button_run_prophet.on_click(button_run_prophet_callback)

# Text: Display various information on status
# (on startup: Newest event date in ACLED database)
text_status = Div(text=get_newest_date_in_db_text())

# Create widgetbox with all above widgets
widgets_prophet = widgetbox(text_prophet_parameters, checkbox_log_scale,
                            slider_window_size, text_reg_changepoint,
                            text_reg_season, slider_freq_days, text_periods,
                            button_run_prophet, text_status)


# Below: Build complete plot from all above definitions

# Build figure: Map of Africa
africa_map_figure_title = "Color map of accumulated reported events in selected time period"
africa_map = africa_map_figure(africa_map_cds, map_select_area_cds,
                               data='value',
                               title=africa_map_figure_title,
                               text_font_size="18px",
                               africa_map_plot_width=africa_map_plot_width)
# Add colormap legend
colormap_legend = africa_map_add_legend(palette, africa_map_plot_width)
map_and_colormap_legend = column(children=[africa_map, colormap_legend],
                                 sizing_mode='fixed')
# Add controls
fig_map_and_controls = row(map_and_colormap_legend,
                           column(children=[widgets_dataset, widgets_prophet]))

# Build figure: Last result of Prophet model
fig_prophet_result = figure(x_axis_type='datetime',
                            title="Result of last Prophet Time Series model",
                            plot_height=600, plot_width=1100)
fig_prophet_result.xaxis[0].axis_label = 'Time'
fig_prophet_result.yaxis[0].axis_label = '{} in time window'.format(variable.capitalize())
fig_prophet_result.title.text_font_size = "18px"
fig_prophet_result.grid.grid_line_alpha = 0.7
fig_prophet_result.x_range.range_padding = 0
fig_prophet_result.patch('x', 'y', legend='Uncertainty', source=prophet_cds_bband_uncertainty,
                         color='#7570B3', fill_alpha=0.1, line_alpha=0.3)
fig_prophet_result.line('x', 'y', legend='Fitted model', source=prophet_cds_y_hat_fit,
                        line_width=2, line_alpha=0.6)
fig_prophet_result.scatter('x', 'y', legend='Training data', source=prophet_cds_y_train)
fig_prophet_result.scatter('x', 'y', legend='Test data', source=prophet_cds_y_test, color='red')

# Populate data sources for first update
update_time_window_datasources()

# Add layout to document root
curdoc().add_root(column(fig_map_and_controls, fig_prophet_result))
