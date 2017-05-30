ACLED Data Science Laboratory
==============================

This repository contains the the result of the project assignment in 'STK-INF4000 Selected Topics in Data Science' at the University of Oslo, spring 2017.

The project uses the ACLED data set. From acleddata.com:
“ACLED (Armed Conflict Location & Event Data Project) is the most comprehensive public collection of political violence and protest data for developing states.“

It is updated on a weekly basis and contains approximately 156.000 entries from 1997 up until today.

The project
-----------

The project goal was to help users of the ACLED dataset better understand (and possibly act on the information in the data, e.g. if the user is an aid organization).

Specifically, we have created a tool that provides a graphical user interface that visualizes the data and encourages data exploration. The tool has a pipeline to Prophet, a time series prediction package created by Facebook (link below).  

Application GUI:
![Image of ACLED data science application GUI](/img/ACLED_application_screenshot.png "ACLED data science application")

Once the user has selected an area of interest (both time and space), the corresponding data is passed to Prophet, that tries to create a model that predicts the future trend.

Once the parameters are set and the user of the application has found areas of interest in the dataset, the parameters could be saved for further analysis and parameter optimization using Prophet.

The pipeline for saving presets and templates for further analysis has not been implemented, but the provided notebook on this repository illustrates how this would work in principle.

Technical overview
------------------

The dataset is downloaded using the ACLED API (http://www.acleddata.com/data/acled-api/) and stored to a MongoDB database on the host system, using PyMongo.

An ESRI shapefile from http://www.naturalearthdata.com/ is used with Geopandas to extract the contour data of the countries in Africa.

The graphical user interface is created using Bokeh and served from a Bokeh server.

Prophet (https://facebookincubator.github.io/prophet/) "... is a forecasting procedure implemented in R and Python." In addition to the automatic optimization ran inside our application, expert knowledge (e.g. changepoints and periodic events) can be added to the model, further improving the results.

Disclaimer
----------

The work in this repository is to be considered a 'proof of concept' of a data science application on the ACLED data set and is not an implemented and live service.
