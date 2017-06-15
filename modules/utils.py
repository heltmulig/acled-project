"""Utility functions for the application.

Currently only one function:

"""

def pivot_resampled_filtered(df_full, events_to_include, params_pivot):
    """ Returns pivot table.

    Params:
    df_full: All ACLED data points
    events_to_include: List of event types to be included in pivot
    params_pivot: Parameters passed to the pandas pivot function (dict)
    resample_period: Resample period of pivot table (string on format '1M')
    """

    df_temp = df_full.copy()

    if events_to_include == []:
        df_piv = df_temp.pivot_table(**params_pivot)
        df_piv[:] = 0
    else:
        mask = df_temp['event_type'].isin(events_to_include)
        df_temp = df_temp[mask]
        df_piv = df_temp.pivot_table(**params_pivot)

    return df_piv

def resample_pivot_table(df_piv, resample_period='1M'):
    """ Takes pivot table, returns resampled pivot table

    Params:
    df_piv: Pivot table, output of 'pivot_resampled_filtered'.
    resample_period: Resample period of pivot table (string on format '1M')
    """
    df_resampled = df_piv.resample(resample_period,
                             closed='left',
                             label='right'
                            ).sum().fillna(value=0).T

    return df_resampled
