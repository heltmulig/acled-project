"""Example presets for ACLED Data Science Laboratory."""

presets = [
{
    'name': "Somalia 2012-2014",
    'current_month': 208,
    'acc_months': 29,
    'selected_area' : [41, 51.5, -2, 12],   # order: x1, x2, y1, y2
    'log_of_y': False,
    'time_window': 7,
    'reg_changepoint': 0.05,
    'reg_season': 10.0,
    'freq_days': 1,
    'periods': 365
},
{
    'name' : "Nigeria 2013-2015",
    'current_month': 221,
    'acc_months': 29,
    'selected_area' : [1.7, 15.4, 3.7, 14.7],
    'log_of_y': False,
    'time_window': 7,
    'reg_changepoint': 0.2,
    'reg_season': 10.0,
    'freq_days': 1,
    'periods': 365
},
{
    'name' : "Libya 2012-2016 (Difficult example)",
    'current_month': 229,
    'acc_months': 45,
    'selected_area' : [9, 25, 20, 33],
    'log_of_y': False,
    'time_window': 7,
    'reg_changepoint': 1.0,
    'reg_season': 10.0,
    'freq_days': 1,
    'periods': 365
},
{
    'name' : "Sudan 2011-2015 (Parameter issue #1)",
    'current_month': 228,
    'acc_months': 58,
    'selected_area' : [21.8, 39.1, 8.7, 22.3],
    'log_of_y': False,
    'time_window': 7,
    'reg_changepoint': 0.05,
    'reg_season': 10.0,
    'freq_days': 1,
    'periods': 365
},
{
    'name' : "Sudan 2011-2015 (Parameter issue #2)",
    'current_month': 228,
    'acc_months': 58,
    'selected_area' : [21.8, 39.1, 8.7, 22.3],
    'log_of_y': False,
    'time_window': 7,
    'reg_changepoint': 1,
    'reg_season': 10.0,
    'freq_days': 1,
    'periods': 365
}
]

names = [preset['name'] for preset in presets]
