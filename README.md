# refuel

## Overview:
This repository contains the code, visualizations and presentations of my capstone project. This project was done in the last four week during my data science bootcamps. It's purpose is to implement some of the skills learned during the bootcamp and present it on the final event.

## Structure:
    - Project description
    - The data
    - Technologies used in this project
    - Presentations
    - Python code

## Project description:

Gasoline prices are very volatile. Prices can differ more than 20 euro cents during the day. On the other hand, people have  to fuel their cars. Be it the commuter who has to get to work or the parent that brings their children to school. To be independent of fuel prices is very hard and is in most cases connect to increased efforts and substantial investments. This is especially true for companies in the transport sector. Where fueling cost cover a good amount of their total costs. 
There are many solutions on the market that give the customers a good overview of the current fuel prices in the area of interest. But studies have shown that even though a majority of car drivers know of the existence of such tools, only 25% of them actively use such solutions. That is because these apps are not convenient enough. None of the apps on the market offer price predictions let alone fueling advice. 

All german gas stations are required to submit every price change to the antitrust agency in real time. Generously _tankerkoenig.de_ makes this data publicly available under the creative commons license.  
This makes this topic extremely interesting for a data scientist! Publicly available data with perfect accuracy, highly volatile market with great saving potential and a giant potential customer base, just waiting to be tapped:

So here is the big picture for my capstone project:
Predict fuel prices in the future for any given gas station in Germany.  Then predict when a driver needs to fuel the car. Compare the expenses to traditional fueling strategies and calculate the cost savings. Combined with an app this forecast has the potential to be used by a large amount of the current 50 millions car owners. 

## The Data:
The data was downloaded from the _tankerkoenig.de_  azure repository: [Link](https://dev.azure.com/tankerkoenig/_git/tankerkoenig-data)

The are separated into two parts:
    - Gas station data: Contains the actual gas station data: uuid, name, brand, street, house_number, post_code, city, latitude, longitude
    - Price data: All price changes over a day a put into a single csv-file, the features are as follows: date, station_uuid, diesel, e5, e10, dieselchange, e5change, e10change


## Technologies used in this project:
- Data Preprocessing: _pandas_, _numpy_, _scikit-learn_, _scipy_
- Data Visualization: _matplotlib_, _bokeh_ 
    - Geoplots: _folium_, _kepler gl_
- Clustering: _kmeans-clustering_, _gaussian mixture model_ (from _scikit-learn_), _pca_
- Time Series Predictions: _SARIMA_ (from _statsmodels_)

## Presentations:
The directory __presentations__ contains all performed presentations during the final event, with their scripts in the notes. 
    - __One_minute_pitch.pdf__: The name says it all!
    - __refuel_business_deck.pdf__: Presentation show to people coming to my desk on the event. - Not scripted -
    - __refuel_presentation_3_minutes.pdf__:Presentations performed in the evening part of the final event on stage. - scripted -  

## Python Code:
All operations used more frequently have been implemented as functions. These functions are all in the __refuel_tools.py__ script. All notebooks import this file and use its functions:

### EDA and time series visualizations with geo information:
The idea was to visualize the price changes and visualize them on a greater scale so other market assumptions could be made. This has been done once using __folium__ and once using __kepler gl's__ ipython friendly implementations. The resulting plots can be found in the __plots__ directory:

- __time_series_geoplot.ipynb__: Contains the folium implementation

- __kepler_time_series_visualization.ipynb__: Contains the kepler gl implementation

### Clustering: 
The goal in this section was to create a number of different statistical features as a base for later clustering. I was able to separate three different consistent cluster groups only based on their price changes. 

-   __make_clustering_features.ipynb__: This notebooks produces statistical features.

- __clustering.ipynb__ Clustering the created features with gmm, and kmeans

- __make_clustering.ipynb__ Plot the results of the clustering.

### refuel:
This notebooks contain the processes connected to __refuel__ idea:

- __make_driving_profile.ipynb__: Here a driving profile of a commuter is created. Which will be used as baseline to compared the impact of the refuel-algorithm

- __refuel.ipynb__: This notebook contains the heart of this project. Here the forecast are created. And than evaluated in a real-world-like example.

apstone Project
