import numpy as np
from fbprophet import Prophet
import pandas as pd

# For clustering:
from datetime import datetime, timedelta, date


def prophet_prepare_df_cols(df, x_range, y_range, date_range,
                            days_to_validate, log_of_y=False, time_window=7,
                            pivot_aggr_fn='count', date_col='event_date',
                            y_col='fatalities'):
    """ Returns dataframe with columns ['ds', 'y'], with points in right half-open interval [start_date, end_date)

    Params:
    df: pandas dataframe with two columns [date, value] for analysis
    date_col: Name of date column
    y_col: Name of value/y column

    pivot_aggr_fn: Aggregation fn used in pivot. E.g.'count' (# of items) or 'sum' (sum of items)
    start_date: Begininning of period to include (datetime.time object)
    end_date: End of period to include (datetime.time object)
    time_window: Integer signifying the number of days/size of sliding window to be used
    log_of_y: Uses log(df[y_col]) if set to true

    Returns
    df_train: all training points inside specified time window.
    df_val: validation points, starting from t=end_date - time_window
    """
    start_date = date_range[0]
    end_date = date_range[1]

    time_window_str = str(time_window) + 'd'
    days_to_validate_str = str(days_to_validate) + 'd'

    # Filter data points inside the defined bounding box:
    mask_lon = (df['longitude'] >= x_range[0]) & (df['longitude'] <= x_range[1])
    mask_lat = (df['latitude'] >= y_range[0]) & (df['latitude'] <= y_range[1])

    df_masked = df.loc[(mask_lon & mask_lat)]

    # Create pivot table, using selected aggregate function:
    df_piv = df_masked.pivot_table(index=date_col,
                                    values=[y_col],
                                    aggfunc=pivot_aggr_fn)

    if log_of_y:
        df_piv[y_col] = np.log(df_piv[y_col])

    # Training data - Mask to the selected time frame:
    date_mask_train = (df_piv.index >= start_date) & (df_piv.index < end_date)
    df_train = df_piv.loc[date_mask_train]

    # Validation data - Includes points after 'end_date'
    start_date_val = pd.Timestamp(end_date) - pd.Timedelta(time_window_str)
    end_date_val = pd.Timestamp(end_date) + pd.Timedelta(days_to_validate_str)

    date_mask_val = (df_piv.index >= start_date_val) & (df_piv.index < end_date_val)
    df_val = df_piv.loc[date_mask_val]

    # Reindex and rename to Prophet format:
    df_train = df_train.reset_index(level=[0])
    df_train = df_train.rename(columns={date_col:'ds', y_col:'y'})

    df_val = df_val.reset_index(level=[0])
    df_val = df_val.rename(columns={date_col:'ds', y_col:'y'})

    # Create rolling time window:
    if time_window > 1:
        df_train = df_train.rolling(time_window, on='ds').sum()
        df_train = df_train.dropna().reset_index(drop=True)

        df_val = df_val.rolling(time_window, on='ds').sum()
        df_val = df_val.dropna().reset_index(drop=True)

    return df_train, df_val


def prophet_train(df_train, periods=20, freq='2w', prophet_param={}):
    """Fits df using Prophet.

    Params:
    df_train: dataframe to fit
    periods: periods to predict (of freq)
    freq: sample frequency in prediction

    Returns Prophet object.
    """
    m = Prophet(**prophet_param)

    m.fit(df_train)
    future = m.make_future_dataframe(periods, freq)
    forecast = m.predict(future)

    return m, forecast

def run_prophet_plot(m, forecast):
    """Plots trained model using Prophets built in methods.
    This function is not used by Bokeh app, but can be used
    from notebook to analyse further.

    Params:
    m: Prophet object [Where m.predict(*) has been run]
    forecast: result of m.predict(*)
    """
    from matplotlib import pyplot as plt

    m.plot(forecast)

    for cp in m.changepoints:
        plt.axvline(cp, c='gray', ls='--', lw=1)

    m.plot_components(forecast)




    return None

def run_prophet_prediction(preprocess_params, reg_changepoint, reg_season,
                           periods, freq, prophet_plot=False):
    days_to_validate= periods*freq
    freq = str(freq)+'d'

    preprocess_params['days_to_validate'] = days_to_validate

    df_train, df_val = prophet_prepare_df_cols(**preprocess_params)

    prophet_param = {
        'changepoint_prior_scale': reg_changepoint,
        'seasonality_prior_scale': reg_season,
        'interval_width': 0.8, # Standard in Prophet = 0.8
        # With rolling window, weekly seasonality is not usefull:
        'weekly_seasonality': False,
        'yearly_seasonality': 'auto'
    }

    m, forecast = prophet_train(df_train, periods=periods, freq=freq, prophet_param=prophet_param)

    if prophet_plot:
        run_prophet_plot(m, forecast)

    return forecast, df_train, df_val
