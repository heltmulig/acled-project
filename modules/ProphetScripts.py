"""This module handles the interface between the 'ACLED data science laboratory'
application and Prophet.

Intended interface is the function 'run_prophet_prediction' that takes the input
dataframe, masking parameters (time and space) as well as Prophet model
parameters.

Functions:
prophet_prepare_df_cols: Preprocessing of dataframe, including masking
prophet_train:           Runs Prophet training, returns trained model
run_prophet_plot:        Plots model. Note: Cannot be ran from Bokeh application
run_prophet_prediction:  Function taking in dataframe and masking parameters,
                         running preprocessing and training and returns the
                         forecast from the model, training data and test data.
"""
import numpy as np
from fbprophet import Prophet
import pandas as pd

def prophet_prepare_df_cols(df, x_range, y_range, date_range,
                            days_to_test, log_of_y=False, time_window=7,
                            pivot_aggr_fn='count', date_col='event_date',
                            y_col='fatalities'):
    """ Function that preprocesses incoming dataframe, masking according to the
    provided parameters. Returns training and testing dataframes on format
    ['ds', 'y'], where 'ds' corresponds to dates and 'y' the value at 'ds'.

    Params:
    df:         Pandas dataframe with columns [date_col, y_col] (date and value)
    x_range:    Longitude parameters [x_min, x_max]
    y_range:    Latitude paramters [y_min, y_max]
    date_range: Time window, [start_date, end_date], both being datetime objects
    days_to_test: Days into future for the returned df_test
    log_of_y:   Boolean. Returns log(df[y_col]) as value, if true
    time_window: Size of sliding window (each point represents sum of last days)
    pivot_aggr_fn: Accepts values: 'count' or 'sum'. 'count' retuns y as the
                 number of events in the given time window, 'sum' returns the
                 sum of the events in the given time window.
    date_col:    Column where date is stored
    y_vol:       Column values we wish to analyse are stored.

    Returns
    df_train:    All training points inside specified time window.
    df_test:     Test points, from 'end_date' until 'end_date+days_to_test'
    """
    start_date = date_range[0]
    end_date = date_range[1]

    time_window_str = str(time_window) + 'd'
    days_to_test_str = str(days_to_test) + 'd'

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

    # Test data - Includes points after 'end_date'
    start_date_val = pd.Timestamp(end_date) - pd.Timedelta(time_window_str)
    end_date_val = pd.Timestamp(end_date) + pd.Timedelta(days_to_test_str)

    date_mask_val = (df_piv.index >= start_date_val) & (df_piv.index < end_date_val)
    df_test = df_piv.loc[date_mask_val]

    # Reindex and rename to Prophet format:
    df_train = df_train.reset_index(level=[0])
    df_train = df_train.rename(columns={date_col:'ds', y_col:'y'})

    df_test = df_test.reset_index(level=[0])
    df_test = df_test.rename(columns={date_col:'ds', y_col:'y'})

    # Create rolling time window:
    if time_window > 1:
        df_train = df_train.rolling(time_window, on='ds').sum()
        df_train = df_train.dropna().reset_index(drop=True)

        df_test = df_test.rolling(time_window, on='ds').sum()
        df_test = df_test.dropna().reset_index(drop=True)

    return df_train, df_test


def prophet_train(df_train, prophet_param, periods=100, freq='1d'):
    """Fits df using Prophet. Predicts period (periods*freq) into the future,
    Where freq has a frequency unit (e.g. 'd', 'w').

    Params:
    df_train: dataframe to fit
    periods: periods to predict (of freq)
    freq: sample frequency in prediction

    Returns Prophet object.
    """
    model = Prophet(**prophet_param)

    model.fit(df_train)
    future = model.make_future_dataframe(periods, freq)
    forecast = model.predict(future)

    return model, forecast

def run_prophet_plot(model, forecast):
    """Plots trained model using Prophets built in methods.
    This function is not used by Bokeh app, but can be used
    from notebook to analyse data further.

    Params:
    m: Prophet object [Where model.predict(*) has been run]
    forecast: result of model.predict(*), as executed on Prophet object m
    """
    from matplotlib import pyplot as plt

    model.plot(forecast)

    for changepoint in model.changepoints:
        plt.axvline(changepoint, c='gray', ls='--', lw=1)

    model.plot_components(forecast)

    return None

def run_prophet_prediction(preprocess_params, reg_changepoint, reg_season,
                           periods, freq, prophet_plot=False):
    """Interface function towards the ACLED Bokeh application.

    Calls preprocessing function that masks the dataframe according to the
    specified parameters, runs the Prophet training, returning the result.

    Params:
    preprocess_params: Dictionary with preprocessing parameters
    reg_changepoint: Prophet parameter 'changepoint_prior_scale' (float)
    reg_season:      Prophet parameter 'holidays_prior_scale' (float)
    periods:         Period into future Prophet will predict (integer)
    freq:            Frequency per period of above duration (integer, days)
    prophet_plot:    Boolean deciding whether to execute Prophet plots

    Returns:
    forecast:        Forecast from Prophet, list with all forecast parameters
    df_train:        Training data as sent to Prophet
    df_test:         Test data, (periods*freq) days of data directly
                     following df_train. Prophet has not previously seen this
    """
    days_to_test = periods*freq
    freq = str(freq)+'d'

    preprocess_params['days_to_test'] = days_to_test

    df_train, df_test = prophet_prepare_df_cols(**preprocess_params)

    prophet_param = {
        'changepoint_prior_scale': reg_changepoint,
        'seasonality_prior_scale': reg_season,
        'interval_width': 0.8, # Standard in Prophet: 0.8
        'yearly_seasonality': 'auto',
        # With rolling window, weekly seasonality is not usefull:
        'weekly_seasonality': False
    }

    model, forecast = prophet_train(df_train, prophet_param=prophet_param,
                                    periods=periods, freq=freq)

    if prophet_plot:
        run_prophet_plot(model, forecast)

    return forecast, df_train, df_test
