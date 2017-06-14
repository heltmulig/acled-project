"""Bokeh server and session callbacks. Used to load the ACLED database when
the server starts (so it only happens once)."""

import sys
sys.path.insert(0,'modules')

import acleddata
from bokeh.settings import logging as log

def on_server_loaded(server_context):
    log.debug("Loading ACLED dataset ...");
    acled = acleddata.ACLED()
    acled.mongodb_update_database() # Query the ACLED API for new entries
    df_full = acled.mongodb_get_entire_database()

    # Strip and capitalize event_types column to avoid typing differences
    # such as 'Strategic development' and 'Strategic Development ' (trailing space)
    df_full['event_type'] = df_full['event_type'].apply(lambda x: x.strip().capitalize())

    setattr(server_context, 'df_full', df_full)
    log.debug("ACLED dataset loaded.");

def on_session_created(session_context):
    log.debug("Starting session")
    setattr(session_context._document, 'df_full', session_context.server_context.df_full)
