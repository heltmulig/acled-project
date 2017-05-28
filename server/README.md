# Starting server
From repository root, start server with
`bokeh serve --log-level debug server/`

# Connecting
Open this link in web browser
`http://localhost:5006/server`

# Problems
Python `bokeh==0.12.5` plotting package may be problematic (Python process just hangs with cpu burn). However, bokeh commit `0.12.6dev5+50.g1cae46e` from github is known to work.
