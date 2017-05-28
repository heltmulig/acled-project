# Adding modules folder to sys.path:
import sys
sys.path.insert(0, 'modules')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bokeh.settings import logging as log
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, LogColorMapper, Range1d, Plot, Text
from bokeh.models.widgets import Div, CheckboxGroup, TextInput, Select, Button, Slider
from bokeh.models.glyphs import Rect
from bokeh.palettes import OrRd9 as palette
palette.reverse()

from presets import presets as selected_presets
from presets import names as selected_presets_names
from ProphetScripts import run_prophet_prediction


log.debug("STARTED ACLED APPLICATION SESSION.")
curdoc().title = "ACLED Science Laboratory"

# Predict events or fatailities (experimental Prohpet value)
variable = 'events'
if variable == 'fatalities':
    pivot_aggr_fn = 'sum'
else:
    pivot_aggr_fn = 'count'

# Get acled pandas.DataFrame from server context
df_full = curdoc().df_full

from ImportShapefile import ImportShapefile
link_ESRI_shp = 'data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
gpd_df = ImportShapefile(link_ESRI_shp).get_df()

from acled_preprocess import acled_preprocess
df_full, gpd_df = acled_preprocess(df_full, gpd_df)

df_piv = df_full.pivot_table(index='event_date',
                              columns='country',
                              values='fatalities',
                              aggfunc=pivot_aggr_fn)

# Aggregate over time window:
time_window = '1M'
df_piv = df_piv.resample(time_window,
                         closed='left',
                         label='right'
                        ).sum().fillna(value=0).T

gpd_df['value'] = df_piv.iloc[:, 0]

# INCOMPATIBILITY FIX
# Angola is a MultiPolygon (two unconnected areas on land). Geopandas
# separates these two polygons by inserting np.float(np.nan) in the
# coordinate list. This type will cause a ValueError exception in
# bokeh, stating the type is not JSON serializable.
gpd_df.set_value('Angola', 'x', gpd_df.loc['Angola', 'x'][:66])
gpd_df.set_value('Angola', 'y', gpd_df.loc['Angola', 'y'][:66])


# BOKEH GLYPH DATA SOURCES
acled_cds = ColumnDataSource(gpd_df.sort_index())
legend_cds = ColumnDataSource(dict(tick_max=[ '{:.0f}'.format(max(acled_cds.data['value'])) ]))

prophet_cds_y_hat = ColumnDataSource(data=dict(x=[], y=[]))
prophet_cds_y = ColumnDataSource(data=dict(x=[], y=[]))
prophet_cds_y_val = ColumnDataSource(data=dict(x=[], y=[]))

prophet_b_band = ColumnDataSource(dict(x=[], y=[]))

# Selection box parameters
'''
  (0) ----------- (1)    (0) = (x0,y1)
     |           |       (1) = (x1,y1)
     |           |       (2) = (x1,y0)
     |           |       (3) = (x0,y0)
  (3) ----------- (2)
'''
x_min = -18
y_max = 38
y_min = -36
x_max = 52

x_range = [16,35]
y_range = [-15,10]

source_prophet = ColumnDataSource(
           dict(x=[[x_range[0], x_range[1], x_range[1], x_range[0]]],
                y=[[y_range[1], y_range[1], y_range[0], y_range[0]]],
                name=['prophet_select']))


def bokeh_add_legend(palette, plot_dim):
    # Adds glyphs for legend. Based on:
    # https://github.com/chdoig/scipy2015-blaze-bokeh/blob/master/1.5%20Glyphs%20-%20Legend%20(solution).ipynb

    # Set ranges
    xdr = Range1d(0, plot_dim)
    ydr = Range1d(0, 60)

    # Create plot
    legend = Plot(
        x_range=xdr,
        y_range=ydr,
        plot_width=plot_dim,
        plot_height=60,
        min_border=0,
        toolbar_location=None,
        outline_line_color="#FFFFFF",
    )

    # Add the legend color boxes and text:
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
    tick_max = Text(x=width*(len(palette)+0.5), y=0, text='tick_max')
    legend.add_glyph(legend_cds, tick_max)

    return legend


def bokeh_plot_map(bokeh_cds, source_prophet, data='value', title="Map with no title",
                   text_font_size=None, color_mapper_type='log', plot_dim=700):
    """ Finalize docstring (TODO)
    Plotting map contours

    Input:
    bokeh_cds - ColumnDataSource including columns
        'name': Country name
        'x' and 'y' that cointain exterior coordinates of the contries contours.
        One column with values used for colouring (see 'data')

    data: String containing name of column in 'df' to fill contours
    """
    if color_mapper_type=='log':
        color_mapper = LogColorMapper(palette=palette)
    elif color_mapper_type=='linear':
        color_mapper = LinearColorMapper(palette=palette)

    TOOLS = "pan,wheel_zoom,reset,save"

    fig = figure(title=title, tools=TOOLS,
               plot_width=plot_dim, plot_height=plot_dim,
               active_drag=None)
    fig.xaxis[0].axis_label = 'Longitude'
    fig.yaxis[0].axis_label = 'Latitude'
    if text_font_size is not None:
        fig.title.text_font_size = text_font_size

    contour = fig.patches('x', 'y', source=bokeh_cds,
                        fill_color={'field': data, 'transform': color_mapper},
                        fill_alpha=1, line_color="black", line_width=0.3)

    hover = HoverTool(renderers=[contour])
    hover.tooltips=[("Country", "@name"),
                    (data.capitalize(), "@"+data)]

    pglyph = fig.patches('x', 'y', source=source_prophet, color=["navy"],
                       alpha=[0.3], line_width=2)

    fig.add_tools(hover)

    return fig


def prophet_to_plot(forecast, df_train, df_val):
    """Input to main.py:

    forecast: Data from trained prophet model

    df_train: Raw data used in training
    df_val:   Raw data for validation (same preprocessing as for df_train)
    """

    # Define Bollinger Bands.
    upperband = forecast['yhat_lower']
    lowerband = forecast['yhat_upper']
    x_data = forecast['ds']

    prophet_cds_y_hat.data = dict(x=forecast['ds'], y=forecast['yhat'])
    prophet_cds_y.data = dict(x=df_train['ds'], y=df_train['y'])
    prophet_cds_y_val.data = dict(x=df_val['ds'], y=df_val['y'])
    prophet_b_band.data['x'] = np.append(x_data, x_data[::-1])
    prophet_b_band.data['y'] = np.append(lowerband, upperband[::-1])

    # Bollinger shading glyph:
    #band_x = np.append(x_data, x_data[::-1])
    #band_y = np.append(lowerband, upperband[::-1])


def get_prophet_debug_text():
    return "Status: Coordinates {}, {}, {}, {}".format(
        slider_x1.value, slider_y1.value,
        slider_x2.value, slider_y2.value)

def get_period_text():
    end_of_period = slider_et.value  # current_month
    period_size = slider_ws.value    # prev. months to accumulate
    start_of_period = max(1, end_of_period - period_size)
    log.debug('(start_of_period=%d,end_of_period=%d)' % (start_of_period, end_of_period))
    return "<table><tr><td>Start of period:</td><td>{}</td></tr><tr><td>End of period (including):</td><td>{}</td></tr></table>".format(
        df_piv.columns[start_of_period-1].strftime("%Y-%m"),
        df_piv.columns[end_of_period-1].strftime("%Y-%m"))

'''
def get_start_index():
    val = max(0, slider_et.value - slider_ws.value)
    return val

def get_end_index():
    val = min(number_of_months, slider_et.value)
    return val

def slider_et_callback(attrname, old, new):
    """Current month"""
    #log.debug('slider_et_callback (value=%d)' % slider_et.value)
    #import pdb; pdb.set_trace()
    #end_of_period = slider_et.value
    #period_size = slider_ws.value
    #start_of_period = max(0, end_of_period - period_size)
    # Upate new range of 'months to sum' slider
    slider_ws.update(end=min(get_start_index(), get_end_index()))
    slider_ws.value = min(1, slider_ws.value, slider_et.value)
    #slider_ws.value = min(period_size, end_of_period)
    text_period.text = get_period_text()
    log.debug('slider_et_callback (value=%d) (start=%d,end=%d)' % (slider_et.value, get_start_index(), get_end_index()))
    acled_cds.data['value'] = df_piv.iloc[:, get_start_index():get_end_index()].sum(axis=1)
    legend_cds.data['tick_max'] = ['{:.0f}'.format(max(acled_cds.data['value']))]

def slider_ws_callback(attrname, old, new):
    """Months to accumulate"""
    #log.debug('slider_ws_callback (value=%d)' % slider_ws.value)
    #end_of_period = slider_et.value
    #period_size = slider_ws.value
    #start_of_period = max(0, end_of_period - period_size)
    text_period.text = get_period_text()
    log.debug('slider_ws_callback (value=%d) (start=%d,end=%d)' % (slider_et.value, get_start_index(), get_end_index()))
    #if start_of_period == 0: import pdb; pdb.set_trace()
    acled_cds.data['value'] = df_piv.iloc[:, get_start_index():get_end_index()].sum(axis=1)
    legend_cds.data['tick_max'] = ['{:.0f}'.format(max(acled_cds.data['value']))]
'''

def slider_ws_callback(attrname, old, new):
    #log.debug('slider_ws_callback (value=%d)' % slider_ws.value)
    text_period.text = get_period_text()
    end_of_period = slider_et.value
    period_size = slider_ws.value
    start_of_period = max(0, end_of_period - period_size)
    log.debug('slider_ws_callback (value=%d) (start=%d,end=%d)' % (slider_et.value,start_of_period,end_of_period))
    #if start_of_period == 0: import pdb; pdb.set_trace()
    acled_cds.data['value'] = df_piv.iloc[:, start_of_period:end_of_period+1].sum(axis=1)
    legend_cds.data['tick_max'] = ['{:.0f}'.format(max(acled_cds.data['value']))]

def slider_et_callback(attrname, old, new):
    #log.debug('slider_et_callback (value=%d)' % slider_et.value)
    text_period.text = get_period_text()
    end_of_period = slider_et.value
    period_size = slider_ws.value
    start_of_period = max(0, end_of_period - period_size)
    # Upate new range of 'months to sum' slider
    slider_ws.update(end=end_of_period)
    slider_ws.value = min(slider_ws.value, end_of_period)
    log.debug('slider_et_callback (value=%d) (start=%d,end=%d)' % (slider_et.value,start_of_period,end_of_period))
    #acled_cds.data['value'] = df_piv.iloc[:, slider_et.value]
    #import pdb; pdb.set_trace()
    ####################
    #if start_of_period == 0: import pdb; pdb.set_trace()
    acled_cds.data['value'] = df_piv.iloc[:, start_of_period:end_of_period+1].sum(axis=1)
    legend_cds.data['tick_max'] = ['{:.0f}'.format(max(acled_cds.data['value']))]


# Example plot of Africa:
plot_dim = 850
color_mapper_type='log'

# Creating map:
p = bokeh_plot_map(acled_cds, source_prophet, data='value',
        title="Color map of accumulated reported events in selected time period",
        text_font_size = "18px",
        color_mapper_type=color_mapper_type,
        plot_dim=plot_dim)

# Adding legend:
#legend_params = {'color_mapper_type': color_mapper_type,
#                 'min': min(acled_cds.data['value']),
#                 'max': max(acled_cds.data['value'])
#                 }
legend = bokeh_add_legend(palette, plot_dim)

text_dataset = Div(text='<h3>Area and time parameters</h3>')

select_preset = Select(title="Selected presets", value="None", options=["None"] + selected_presets_names)
def select_preset_callback(attr, old, new):
    try:
        i = selected_presets_names.index(new)
    except ValueError:
        return
    slider_et.value = selected_presets[i]['current_month']
    slider_ws.value = selected_presets[i]['acc_months']
    slider_x1.value = x_min; slider_x2.value = x_max
    slider_y1.value = y_min; slider_y2.value = y_max
    slider_x1.value = selected_presets[i]['selected_area'][0]  # X-Left
    slider_x2.value = selected_presets[i]['selected_area'][1]  # X-Right
    slider_y1.value = selected_presets[i]['selected_area'][2]  # Y-lower
    slider_y2.value = selected_presets[i]['selected_area'][3]  # Y-upper
    checkbox_log.active = [0] if selected_presets[i]['log_of_y'] else []
    slider_window_size.value = selected_presets[i]['time_window']
    text_reg_changepoint.value = str(selected_presets[i]['reg_changepoint'])
    text_reg_season.value = str(selected_presets[i]['reg_season'])
    slider_freq_days.value = selected_presets[i]['freq_days']
    text_periods.value = str(selected_presets[i]['periods'])

select_preset.on_change('value', select_preset_callback)


# Slider indicating end month of time window:
number_of_months = df_piv.shape[1]
slider_et = Slider(start=1, end=number_of_months, value=1, step=1,  # current month
                   title="Current period (month)")
slider_et.on_change('value', slider_et_callback)

# Slider indicating window size:
slider_ws = Slider(start=1, end=slider_et.value, value=1, step=1,   # acc months
                   title="Previous months to accumulate (incl. current)")
slider_ws.on_change('value', slider_ws_callback)


text_period = Div(text=get_period_text())

slider_x1 = Slider(start=x_min, end=x_max, value=source_prophet.data['x'][0][0], step=0.1, title="X-left")
slider_y1 = Slider(start=y_min, end=y_max, value=source_prophet.data['y'][0][2], step=0.1, title="Y-lower")
slider_x2 = Slider(start=x_min, end=x_max, value=source_prophet.data['x'][0][1], step=0.1, title="X-right")
slider_y2 = Slider(start=y_min, end=y_max, value=source_prophet.data['y'][0][0], step=0.1, title="Y-upper")
def slider_callback(attr, old, new):
    global x_range, y_range
    slider_x1.value = min(slider_x1.value, slider_x2.value)
    slider_y1.value = min(slider_y1.value, slider_y2.value)
    slider_x2.value = max(slider_x1.value, slider_x2.value)
    slider_y2.value = max(slider_y1.value, slider_y2.value)
    x_range = [slider_x1.value, slider_x2.value]
    y_range = [slider_y1.value, slider_y2.value]
    text_debug.text = get_prophet_debug_text()
    source_prophet.data['x'] = [[x_range[0], x_range[1], x_range[1], x_range[0]]]
    source_prophet.data['y'] = [[y_range[1], y_range[1], y_range[0], y_range[0]]]


slider_x1.on_change('value', slider_callback)
slider_y1.on_change('value', slider_callback)
slider_x2.on_change('value', slider_callback)
slider_y2.on_change('value', slider_callback)

widgets_dataset = widgetbox(text_dataset, select_preset, slider_et, slider_ws,
        text_period, slider_x1, slider_x2, slider_y2, slider_y1)


# PROPHET WIDGETS
text_prophet_parameters = Div(text='<h3>Prophet parameters</h3>')
checkbox_log = CheckboxGroup(labels=['Logarithmic scale'], active=[])
slider_window_size = Slider(start=1, end=21, value=1, step=1,
                   title='Sliding window size')
text_reg_changepoint = TextInput(value='1.0', title='Regularizer changepoint (decimal number)')
text_reg_season = TextInput(value='1.0', title='Regularizer season (decimal number)')
slider_freq_days = Slider(start=1, end=21, value=1, step=1,
                    title='Frequency of prediction (days in periods)')
text_periods = TextInput(value='1', title='Number of periods to predict (integer)')

# Calculate button
button_prophet = Button(label="Run Prohpet!")
def button_callback():
    global x_range, y_range
    log.debug("Prophet predict!")
    end_of_period = slider_et.value
    period_size = slider_ws.value
    start_of_period = max(0, end_of_period - period_size)
    try:
        reg_changepoint = float(text_reg_changepoint.value)
        reg_season = float(text_reg_season.value)
    except ValueError:
        text_debug.text = 'Regularizer: Not a decimal number!'
        return
    prophet_preprocessing_params = {
         'df': df_full,
         'x_range' : x_range,
         'y_range' : y_range,
         'date_range' : [df_piv.columns[start_of_period], df_piv.columns[end_of_period]],
         'log_of_y': 0 in checkbox_log.active,
         'time_window': slider_window_size.value,
         'pivot_aggr_fn': pivot_aggr_fn }
    try:
        periods = int(text_periods.value)
    except ValueError:
        text_debug.text = 'Number of periods: Not a valid integer!'
        return
    text_debug.text = 'Running Prophet ...'
    try:
        start_time = datetime.now()
        prophet_to_plot(*run_prophet_prediction(
                            prophet_preprocessing_params, reg_changepoint, reg_season,
                            periods=periods, freq=slider_freq_days.value))
        end_time = datetime.now()
        text_debug.text = 'Prophet completed in {:d} seconds'.format((end_time - start_time).seconds)
    except KeyError as e:
        text_debug.text = 'Prophet error: No events within selection box {}'.format(str(e))
    except Exception as e:
        text_debug.text = 'Prophet error: {}'.format(str(e))

button_prophet.on_click(button_callback)
text_debug = Div(text=get_prophet_debug_text())

widgets_prophet = widgetbox(text_prophet_parameters, checkbox_log, slider_window_size,
                    text_reg_changepoint, text_reg_season, slider_freq_days, text_periods, button_prophet, text_debug)

# Building the overall plot:
p_with_legend = column(children=[p, legend], sizing_mode='fixed')
fig_africa = row(p_with_legend, column(children=[widgets_dataset, widgets_prophet]))
fig_prophet = figure(x_axis_type='datetime', title="Result of last Prophet Time Series model",
                    plot_height = 600, plot_width = 1100)
fig_prophet.xaxis[0].axis_label = 'Time'
fig_prophet.yaxis[0].axis_label = '{} in time window'.format(variable.capitalize())
fig_prophet.title.text_font_size = "18px"
fig_prophet.grid.grid_line_alpha = 0.7
fig_prophet.x_range.range_padding = 0
fig_prophet.patch('x', 'y', source=prophet_b_band, color='#7570B3', fill_alpha=0.1, line_alpha=0.3)
fig_prophet.line('x', 'y', source=prophet_cds_y_hat, line_width=2, line_alpha=0.6)
fig_prophet.scatter('x', 'y', source=prophet_cds_y)
fig_prophet.scatter('x', 'y', source=prophet_cds_y_val, color='red')

curdoc().add_root(column(fig_africa, fig_prophet))
