"""Just a few unittests to show the concept. TODO: Add more.
Manually execute from project root:

   pytest tests/

"""
import sys
sys.path.insert(0, 'modules')
from datetime import datetime

import pandas as pd

import acleddata as acled

db = acled.ACLED()

def test_get_entire_database():
    df = db.mongodb_get_entire_database()
    assert type(df) is pd.DataFrame, "Should return pandas.DataFrame"

def test_last_event_date():
    date = db.mongodb_get_newest_event_date()
    assert type(date) is datetime, "Should return datetime.datetime"

def test_get_query():
    starttime = datetime(2016, 1, 1)
    endtime = datetime(2017, 1, 1)
    query = {'event_date': {'$gt':starttime, '$lt':endtime}}
    df = db.mongodb_get_query(query)
    assert type(df) is pd.DataFrame, "Should return pandas.DataFrame"
    assert df.size > 450000, "Database should contain more than 450000 events in the year 2016."
