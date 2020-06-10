"""Refuel_tools is a library containing functions used in the capstone project."""

from os import listdir
import pandas as pd
from scipy.spatial import cKDTree
from math import pi
import numpy as np
import statsmodels.api as sm
import itertools
from scipy.spatial import cKDTree


def load_data_from_csv(timespan=365, 
                       path=r'./tankerkoenig-data/prices/', 
                       verbose=False, 
                       compression=False, 
                       save_as_pickle=True
                      ):
    """
    This function loads the raw data from the csv files. Every day is in a single csv file, 
    summed up by months and years. This function takes days as input to reduce the length of 
    the dataframe. Meaning if days = 365, the produced dataframe will span over the last year 
    of the entire dataset for the last day backwards. 
    
    Make sure you are in the correct working directory --> 
    '/Users/gabriel/nf-ds/Capstone'. 
    
    Finally this funciton 
    saves the produced dataframe as a pickle with xz compression in the data directory.
    
    INPUT: timespan = [int] number of days backwards from the last day as csv file
           path = [string] path to lowest csv directory
           verbose = [bool] produce output while generating the dataframe
           compression = [bool] use compression on generated dataframe, (xz)
           save_as_pickle = [bool] save produced dataframe as pickle
           
    OUTPUT: df = [dataframe], with datetimeindex. at the same time a copy of the generated 
                df is saved as a pickle in the directory ./data/ with the filename df_FULL_{} 
                ending with the timespan.
    """
    
    # ignore hidden files starting with .(dot)
    years = sorted([f for f in listdir(path) if not f.startswith('.')])
    all_days = []

    for year in years:
        months = sorted([f for f in listdir(path + year) if not f.startswith('.')])
        for month in months:
            if month[0] != '.':
                days = sorted(listdir(path + year + '/'+ month))
            for day in days:
                all_days.append(path + year + '/' + month + '/' + day)

    df = pd.DataFrame()
    for filename in all_days[-timespan:]:
        if verbose:
            print('Appending file: {}'.format(filename))
        df_i = pd.read_csv(filename, index_col=None, header=0)
        df_i['date'] = pd.to_datetime(df_i.date, utc=True)
        df_i.set_index('date', inplace=True)
        df = df.append(df_i)
    
    if save_as_pickle:
        if compression:
            df.to_pickle("./data/df_FULL_{}.pkl".format(timespan), compression='xz')
        else:
            df.to_pickle("./data/df_FULL_{}.pkl".format(timespan))
    return df


def get_station_uuid_df(df_stations, 
                        position, 
                        k=1):
    """
    This function takes the df_stations dataframe which contains all the information on the gas stations and 
    a latitude and logitude as list of list. It outputs a dataframe with station and their uuids with the 
    corresponding closest gas stations, sorted by closest. The function also output the radial distance to 
    the given input position.
    
    Input: df_stations = [dataframe] dataframe containing the informations on the gas stations
           position = [float(lat), float(long)] positions to search around
           k = [int] number of neighbors
           
    Output: df_uuid = [dataframe] dataframe containing information on each gas station and a radial #
                      distance to the given position
    """
    
    # Building a tree from all the given stations
    tree = cKDTree(np.c_[df_stations.latitude.ravel(), df_stations.longitude.ravel()])

    distance, index = tree.query([position[0], position[1]], k=k)
    # Have to include an exception for k == 1, because index on the dataframe can't be an integer.
    if k == 1:
        df_uuid = pd.DataFrame(index=[index])
        df_uuid['city'] = df_stations.loc[index, 'city']
        df_uuid['uuid'] = df_stations.loc[index, 'uuid']
        df_uuid['latitude'] = df_stations.loc[index, 'latitude']
        df_uuid['longitude'] = df_stations.loc[index, 'longitude']
        df_uuid['brand'] = df_stations.loc[index, 'brand']
    else:
        # Write other values to dataframe
        df_uuid = pd.DataFrame(index=index)
        df_uuid['city'] = df_stations.loc[index, 'city'].values
        df_uuid['uuid'] = df_stations.loc[index, 'uuid'].values
        df_uuid['latitude'] = df_stations.loc[index, 'latitude'].values
        df_uuid['longitude'] = df_stations.loc[index, 'longitude'].values
        df_uuid['brand'] = df_stations.loc[index, 'brand'].values
    # Convert degree into kilometers
    df_uuid['distance'] = (distance*6371*pi)/180.0
    
    return df_uuid

def get_uuids_around_positions(df_stations, 
                               positions, 
                               k=3):
    """ This function is an extension of "get_station_uuid_df". It uses it as a base but also adds the 
    position of the origin. It was used for creating the evaluation.
    
    INPUT: 
        df_stations = [dataframe] dataframe with the informations on the entire gas station network
        positions = [list of two floats] (latitude, longitude)
        k = [int] number of neighbors to find around the origin
        
    OUTPUT: 
        positions_uuids = [dataframe] with information on the gas station of interest.
    
    """
    positions_uuids = pd.DataFrame()
    for position in positions:
        df_uuid = get_station_uuid_df(df_stations, position=position, k=k)
        df_uuid['origin_lat'] = position[0]
        df_uuid['origin_long'] = position[1]
        positions_uuids = positions_uuids.append(df_uuid)
    return positions_uuids



def make_timeseries(df,
                    df_uuid,
                    cutoff_days=0,
                    sample_rate='5min',
                    add_color=False,
                    verbose=False):
    
    """ 
    The purpose of this function is to convert the raw price data into a resampled time series. It also filters
    the raw dataframe on the uuids given in df_uuid.
    
    INPUT: df = [dataframe] contains the entire raw data created with "load_data_from_csv"
           df_uuid = [dataframe] contains the uuids of interest created with "get_station_uuid_df"
           cut_off_day = [int] how many days to cut of the data in "df" to make sure only entire days are in the
                         final dataframe. Zero should be fine. Since usually the csv contain only entire days.
           add_color = [bool] add color code based on the price for visualization with folium
           verbose = [bool] print information during run
    
    OUTPUT: df_out = [dataframe] resampled dataframe containing only the gas stations given in df_uuid
    """
    df_out = pd.DataFrame()
    lenght = df_uuid.shape[0]
    i = 0
    for _, row in df_uuid.iterrows():
        i += 1
        if verbose:
            print('Working on the {}. of {} uuids.'.format(i, lenght))
            print('Stations uuid = {}'.format(row.uuid))
        
        df_filtered = df[df.station_uuid == row.uuid]
        
        datetime_series = df_filtered.index
        
        # Skip uuids which are empty.
        if len(datetime_series) != 0:

            # get the last day in the dataset, substract additonal day to be sure, that we only have entire days in the df
            last_day = datetime_series[-1] # - pd.Timedelta(days=1)

            # we don't need the column anymore
            df_filtered.drop(['dieselchange', 'e5change', 'e10change'],axis=1,inplace=True)

            # Filtering the dataframe by time_span
            cutoff_date = last_day - pd.Timedelta(days=cutoff_days)
            df_filtered = df_filtered[cutoff_date:last_day]

            # Resampling
            df_resampled = df_filtered.resample(sample_rate).pad()

            # Add position and the uuid
            #df_resampled['uuid'] = row.uuid
            df_resampled['latitude'] = row.latitude
            df_resampled['longitude'] = row.longitude
            #TODO: Implement scaling so coloring can work
            #if add_color:
                #df_resampled['color'] = df_resampled.e5.apply(color_coding)
            #df_resampled.reset_index(inplace=True)
            df_resampled.dropna(inplace=True)
            df_out = df_out.append(df_resampled)
    
    df_out = df_out.rename(columns = {'station_uuid':'uuid'})
    return df_out


def color_coding(poll):
    """
        Function to produce a color for every price bin. This function is not tested entirly, since geoploting 
        was later done with kepler gl.
    """
    bin_edges = np.linspace(0, 1, 10)
    idx = np.digitize(poll, bin_edges, right=True)
    color_scale = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    return color_scale[idx]

def grid_search_SARIMA_parameters(y,
                                  sample_rate='5min',
                                  dim=2):
    """
    This function finds the best SARIMA parameters for a given time series. Depending to the given time
    series the grid search can take hours. A sample_rate below 5min has not been proven successfull.
    Best parameter found are returned as a list - [order, sesonal_parameters]
    
    #### It's very important that the sample_rate given here matches the one later used for predicting! ####
    
    INPUT:
        y = [pandas series] time-series to match the parameter to
        sample_rate = [string] sample rate for arima function
        dim = [int] highest pdq dimension usually not higher than 2.
        
    OUTPUT: 
        List of two tuples, [order, sesonal_parameters] with the best performance
    """
    p = d = q = range(0, dim)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], decompfreq) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    aic_list = []
    para_list = []
    for param in pdq:

        for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(y.values,
                                                order=param,
                                                dates=y.index,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False,
                                                freq=sample_rate
                                           )
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            aic_list.append(results.aic)
            para_list.append([param, param_seasonal])
            #except:
             #   print('failed')
              #  continue
    print('Best Parameter found:')
    print('Seasonal Parameters:{}'.format(para_list[np.argmin(aic_list)][1]))
    print('Order:{}'.format(para_list[np.argmin(aic_list)][0]))
    
    return [para_list[np.argmin(aic_list)][0], para_list[np.argmin(aic_list)][1]]

def make_predictions(df_resampled, 
                     uuids, 
                     fuel_type='e5', 
                     cutoff_day=3, 
                     days_to_predict=3, 
                     add_test_data=True,
                     order=(1,0,1),
                     seasonal_order=(1,0,1,72), 
                     sample_rate='20min'):
    
    """ 
    This function produces a prediction with the given parameters based on df_resampled. For the number of
    days, given in days_to_predict. Cutoff_day and days_to predict should be in most cases be the same.
    
    INPUT:
        df_resampled = [dataframe] Already resampled dataframe containing the time series of all stations of interest.
                    The index of df_resample has to be int.
        uuids = [list] List of uuids for which a forecast should be produced.
        fuel_type = [String] The fuel type of interest. Options: e5, e10, diesel.
        cutoff_day = [Int] of day to cutoff the training data.
        days_to_predict = [Int] Number of day into the future that will be predicted 
        add_test_data = [bool] Switch to turn off adding y_true to the output dataframe
        order = [tuple of three integers] the default values have been the best match for time series with 
        
    OUTPUT:
        df_predictions = [dataframe] containing the forecast.
        
    """
    
    df_predictions = pd.DataFrame()
    
    for uuid in uuids:
        y = df_resampled[df_resampled.uuid == uuid]
        
        # Exeption for dead gas stations
        if len(y) != 0:
            
            # split the dataset into train and test
            last_train_day = y.index[-1].date() - pd.Timedelta(days=cutoff_day)
            y_train = y[y.index < str(last_train_day)]
            y_test = y[y.index >= str(last_train_day)]

            # reduce dataframe to series
            y_train = y_train[fuel_type]
            y_test = y_test[fuel_type]
            
            # drop duplicated rows
            y_train = y_train.loc[~y_train.index.duplicated(keep='first')]

            # Set up SARIMAX model
            mod = sm.tsa.statespace.SARIMAX(y_train,
                                            order=order,
                                            seasonal_order=seasonal_order,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            freq=sample_rate)
            # Model fit
            results = mod.fit()

            # Make Predictions
            y_hat_firstd = str(last_train_day)
            y_hat_lastd = str(last_train_day + pd.Timedelta(days=days_to_predict))
            print('Making Predictions for uuid: {}:'.format(uuid))
            print('From: {}'.format(y_hat_firstd))
            print('Until: {}'.format(y_hat_lastd))
            y_hat = results.get_prediction(start = pd.to_datetime(y_hat_firstd, utc=True),
                                          end = pd.to_datetime(y_hat_lastd, utc=True),
                                          dynamic=False)

            # Write prediction to dataframe
            df_i_predictions = pd.DataFrame(y_hat.predicted_mean, columns=[fuel_type])
            df_i_predictions['uuid'] = uuid
            #df_i_predictions.reset_index(inplace=True)

            if add_test_data: 
                # Write testdata to dataframe
                print('Test Data is added...')
                df_i_true = pd.DataFrame(y_test, columns=[fuel_type])

                # Join the two dataframes
                df_i_predictions = df_i_predictions.join(df_i_true, rsuffix='_test')

            # Append the dataframe from the loop to the main dataframe
            df_predictions = df_predictions.append(df_i_predictions)
        else:
            print('Time series slice was empty!')
    return df_predictions

def number_of_neighbors(row,
                tree,
                radius=1):
    """
    This function calculates the number of neighboring gas station in a given radius.
    Faktor to transform deg to kilometer: r/((6371*pi)/180.0)
    
    #### Before using this function build a cKDtree with the following line:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[df_stations.latitude.ravel(), df_stations.longitude.ravel()])
    ####
    The tree building function was not implemented in this function, because this function
    is used with apply on the entire dataframe.
    
    INPUT:
        row [entire row of dataframe, used with iterrows] - row has to contain lat and long of
                the position.
        tree [cKDTree] - tree with all position from df_stations. This tree need to be build before 
                this function is executed.
        radius = [int] radius in kilometer to look for other gas stations in.
        
    OUTPUT: [int] - Number of neigboring gas stations
    """
    position = [row.latitude, row.longitude]
    neighbors = tree.query_ball_point([position[0], position[1]], radius/((6371*pi)/180.0))
    return (len(neighbors)-1)

def get_ts_features(group):
    """
    This function takes in a groupby object and calulates various statistical values from it.
    
    INPUT: 
        group = [pd groupby object] grouped by the uuid
    OUTPUT:
        various statistical informations as floats
    """
    # Number of prices changes in period
    changes_diesel = group.dieselchange.sum()
    changes_e5 = group.e5change.sum()
    changes_e10 = group.e10change.sum()
    
    # Stats infos
    skewness_diesel = skew(group.diesel)
    skewness_e5 = skew(group.e5)
    skewness_e10 = skew(group.e10)
    
    kurtosis_diesel = kurtosis(group.diesel)
    kurtosis_e5 = kurtosis(group.e5)
    kurtosis_e10 = kurtosis(group.e10)
    
    var_diesel = np.var(group.diesel)
    var_e5 = np.var(group.e5)
    var_e10 = np.var(group.e10)
    
    mean_diesel = np.mean(group.diesel)
    mean_e5 = np.mean(group.e5)
    mean_e10 = np.mean(group.e10)
    
    median_diesel = np.median(group.diesel)
    median_e5 = np.median(group.e5)
    median_e10 = np.median(group.e10)
    
    min_diesel = np.min(group.diesel)
    min_e5 = np.min(group.e5)
    min_e10 = np.min(group.e10)
    
    max_diesel = np.max(group.diesel)
    max_e5 = np.max(group.e5)
    max_e10 = np.max(group.e10)
    
    # Number of price peaks in period
    peaks, _ = find_peaks(group.diesel)
    n_peaks_diesel = len(peaks)
    peaks, _ = find_peaks(group.e5)
    n_peaks_e5 = len(peaks)
    peaks, _ = find_peaks(group.e10)
    n_peaks_e10 = len(peaks)
    
    return changes_diesel, changes_e5, changes_e10, skewness_diesel, skewness_e5, skewness_e10, \
kurtosis_diesel, kurtosis_e5, kurtosis_e10, var_diesel, var_e5, var_e10, mean_diesel, mean_e5, mean_e10, \
median_diesel, median_e5, median_e10, n_peaks_diesel, n_peaks_e5, n_peaks_e10, \
min_diesel, min_e5, min_e10, max_diesel, max_e5, max_e10

def create_geojson_features(df):
    """
    This function takes in a pandas dataframe and creates a dict fitting for the folium plot.
    Input dataframe has to contain the following coloums: longitude, latitude, color
    
    INPUT:
        df = [dataframe] containing the data to plot. 
        
    OUTPUT:
        dict = [dict] in the shape folium wants it.
    """
    features = []
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row.longitude,row.latitude]
            },
            'properties': {
                'time': row.date.__str__()[:-9],
                'style': {'color' : row.color},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': row.color,
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'radius': 7
                }
            }
        }
        features.append(feature)
    return features


def get_true_price(df_ts, 
                   uuid,
                   timestamp,
                   fuel_type='e5'):
    
    """
    ######## This function become obsolet with 'get_prices'#########
    This function takes in an already resampled dataframe. After filtering this dataframe on the
    given uuid it resamples the dataframe with a sample rate of 1 minute and outputs the price.
    This function is necessary because the time series used in the forecast is sampled with a longer
    sample rate. To match the time when the refuel event happends to a price this reresample is 
    necessary. This function was used inside a iterrows loop.
    
    INPUT:
        df_ts = [dataframe] time series
        uuid = [int] the uuid for which the price should be search on
        timestamp = [datetime] time of the fueling event
        fuel_type = [string] type of fuel
        
    OUTPUT:
        true_price [float] - price
    
    """
    # Filter by uuid
    df_filtered = df_ts[df_ts.station_uuid == uuid]
    df_resampled = df_filtered.resample('1min').pad()
    # Filter by date
    true_price = df_resampled[df_resampled.index == timestamp][fuel_type].values
    return true_price

def get_prices(df_forecast, 
               uuid, 
               timestamp,
               fuel_type='e5'):
    """
    This function takes in an already resampled dataframe. After filtering this dataframe on the
    given uuid it resamples the dataframe with a sample rate of 1 minute and outputs the price.
    This function is necessary because the time series used in the forecast is sampled with a longer
    sample rate. To match the time when the refuel event happends to a price this reresample is 
    necessary. This function was used inside a iterrows loop.
    
    INPUT:
        df_ts = [dataframe] time series
        uuid = [int] the uuid for which the price should be search on
        timestamp = [datetime] time of the fueling event
        fuel_type = [string] type of fuel
        
    OUTPUT:
        true_price [list] - predicted price, true price
    
    """
    # Filter df_forecast by the uuid
    df_i_forecast = df_forecast[df_forecast.uuid == uuid]

    if len(df_i_forecast) != 0:
        
    # reresample the time series to get a price for every minute 

        #if not df_i_forecast.index.is_unique:
        df_i_forecast = df_i_forecast.loc[~df_i_forecast.index.duplicated(keep='first')]

        #reresample the time series for find a price for every timestep
        df_i_resampled = df_i_forecast.resample('1min').pad()

        # filter by time stamp
        prices_pred = df_i_resampled[df_i_resampled.index == timestamp][fuel_type].values[0]
        prices_test = df_i_resampled[df_i_resampled.index == timestamp][fuel_type + '_test'].values[0]

    else:
        print('No price for gas station with uuid: {} and timestamp: {} available'.format(uuid, timestamp))
        prices_pred = 0
        prices_test = 0

    return [prices_pred, prices_test]

def make_rolling_forecast(df_ts, 
                          uuids, 
                          first_day_forecast, 
                          last_day_forecast, 
                          cutoff_day=2, 
                          days_to_predict=2, 
                          verbose=False):
    """ 
    This function produces a rolling forecast for the given timespan. It is an extention of make_predictions.
    Forecast is produced for "days_to_predict" in one roll.
    
    INPUT: 
        df_ts = [dataframe] time series resampled. Timespan has to cover the entire forecasting timespan so 
                that test data can also be added.
        uuids = [list] with the uuids a forecast should be produced. Use the following line to convert the 
                df_uuid to a list: uuids = df_uuid.uuid.to_list()
        first_day_forecast = [string] first day the forecast should start eg. '2019-12-08'
        last_day_forecast = [string] last day of the forecast
        cutoff_day = [int] days span for creating test train, should not be bigger than 2
        days_to_predict = [int] days to predict in one roll, should also not be bigger than 2
        verbose = [bool] print output during forecasting
        
    OUTPUT:
        df_forecast = [dataframe] single df with all the forecasting information for the entire timespan and
                      all uuids
                      
        
    """
    first_day_forecast = pd.to_datetime(first_day_forecast, utc=True)
    last_day_forecast = pd.to_datetime(last_day_forecast, utc=True) 
    i_last_day = first_day_forecast
    df_forecast = pd.DataFrame()
    i = 1
    for uuid in uuids:
        if verbose:
            print('')
            print('UUID: {}'.format(uuid))
            print('')
        i_last_day = first_day_forecast
        i = 1
        while i_last_day < last_day_forecast:
            i_last_day = first_day_forecast + pd.Timedelta(days=days_to_predict*i)
            if verbose:
                print('Calculating Forecast until: {}'.format(i_last_day))

            # Cut df_resampled according to current time window
            df_i_resampled = df_ts[df_ts.index < i_last_day]
            if verbose:
                print('last training timestep:     {}'.format(df_i_resampled.index[-1]))
            # Make actual prediction
            df_i_predictions = rt.make_predictions(df_i_resampled, [uuid], cutoff_day=cutoff_day, days_to_predict=days_to_predict, add_test_data=True)
            # Append to output dataframe
            df_forecast = df_forecast.append(df_i_predictions)
            i += 1
    return df_forecast

def calculate_refuel_cost(dp,
                         refuel_treshold=20,
                         refuel_time=10,
                         k=4,
                         benchmark=False,
                         verbose=False):
    """
    This function calculates the refueling cost. Its purpose is to evalute the performance of the 
    price forecast with a real-life refueling algorithm. 
    The basic principle is as follows: Based on the driving profile, if the fuel_vol_end goes below
    the refuel_threshold, a refuel window is opened. That means a price based on the forecast for the
    arrival point and time, AND departure at arrival point and time is calculated. If the rest volume
    is sufficent to reach the next destination two other prices based on the times and position are 
    calculated. Then the cheapest point is choosen. Of course for future calculations the true price is
    used. If benchmark is true, than cost for the benchmark method are calculated: Meaning always refuel 
    at uuid 0, at the start of the journey (0,1,0).
    
    INPUT:
        dp = [dataframe] drivin profile
        refuel_treshold = [int] level when the refueling window is opened
        refuel_time = [int] time is added to the refueling time
        k = [int] number of neigbors considered in each point, has to match the ks in df_forecast,
                  otherwise the no data can be found!
        benchmark = [bool] if True a benchmark is calculated
        verbose = [bool] print infos during calculations
    
    OUTPUT:
        driving_profile = [dataframe] - the inputed dataframe with the added informations
    """


    for i, row in dp.iterrows():
        #TODO: Quickfix, function throws error because in the end in every iteration a new is created. 
        if i == 63:
            break
        if verbose:
            print(i)
        dp.loc[i, 'fuel_vol_end'] = dp.loc[i, 'fuel_vol_start'] - dp.loc[i,'trip_con']
        dp.loc[i+1, 'fuel_vol_start'] = dp.loc[i, 'fuel_vol_end']

        #Dimensions:(uuid, time, position)
        price_pred = np.zeros((k,2,2))
        price_true = np.zeros((k,2,2))


        if dp.loc[i, 'fuel_vol_end'] < refuel_treshold:
            if verbose:
                print('Refuel Window open!')

            # Calculate refueling times
            arrival_pos_0 = dp.loc[i,'start_date'] + pd.Timedelta(minutes=dp.loc[i, 'duration'] + refuel_time)
            start_pos_0 = dp.loc[i+1, 'start_date'] + pd.Timedelta(minutes=refuel_time)
            uuids_pos_0 = df_uuid[df_uuid.origin_lat == dp.loc[i, 'end_lat']].uuid
            j = 0
            for uuid in uuids_pos_0:
                if len(df_forecast[df_forecast.uuid == uuid]) != 0:
                    price_pred[(j,0,0)] = get_prices(df_forecast, uuid, arrival_pos_0)[0]
                    price_pred[(j,1,0)] = get_prices(df_forecast, uuid, start_pos_0)[0]
                    price_true[(j,0,0)] = get_prices(df_forecast, uuid, arrival_pos_0)[1]
                    price_true[(j,1,0)] = get_prices(df_forecast, uuid, start_pos_0)[1]
                    j += 1

            if dp.loc[i+1, 'trip_con'] < dp.loc[i, 'fuel_vol_end']:
                # Will make it to the next position
                arrival_pos_1 = dp.loc[i+1, 'start_date'] + pd.Timedelta(minutes=dp.loc[i+1, 'duration'] + refuel_time)
                start_pos_1 = dp.loc[i+2, 'start_date'] + pd.Timedelta(minutes=refuel_time)
                uuids_pos_1 = df_uuid[df_uuid.origin_lat == dp.loc[i+1, 'end_lat']].uuid

                for uuid in uuids_pos_1:
                    price_pred[(j,0,1)] = get_prices(df_forecast, uuid, arrival_pos_1)[0]
                    price_pred[(j,1,1)] = get_prices(df_forecast, uuid, start_pos_1)[0]
                    price_true[(j,0,1)] = get_prices(df_forecast, uuid, arrival_pos_1)[1]
                    price_true[(j,1,1)] = get_prices(df_forecast, uuid, start_pos_1)[1]
                    j += 1
            price_pred[price_pred == 0] = np.nan
            min_ind = np.unravel_index(np.nanargmin(price_pred, axis=None), price_pred.shape)
            min_prices_pred = price_pred[min_ind]
            min_prices_true = price_true[min_ind]
            benchmark_price = price_true[(0,1,0)]

            ###########
            if benchmark:
                min_prices_true = benchmark_price
                min_ind = (0,1,0)

            if verbose:
                print('Min Index: {}'.format(min_ind))
                print('Minimum predicted price: {}'.format(min_prices_pred))
                print('Minimum true price:      {}'.format(min_prices_true))
                print('Benchmark price:         {}'.format(benchmark_price))
                print('MININUM predictions:')

            ### Smart Refueling:
            if min_ind[0] < k/2:
                # Refuel in pos_0
                pos = dp.loc[i, 'end_lat']
                dp.loc[i, 'predicted_price'] = min_prices_pred
                dp.loc[i, 'true_price'] = min_prices_true
                if min_ind[1] == 0:
                    dp.loc[i, 'refuel_time'] = arrival_pos_0
                else:
                    dp.loc[i, 'refuel_time'] = start_pos_0
                dp.loc[i, 'refuel_vol'] = car_fuel_vol - dp.loc[i, 'fuel_vol_end']
                dp.loc[i, 'refuel_cost'] = dp.loc[i, 'refuel_vol'] * dp.loc[i, 'true_price']
                dp.loc[i, 'fuel_vol_end'] = car_fuel_vol
                dp.loc[i+1, 'fuel_vol_start'] = car_fuel_vol
                if verbose:
                    print('Min price in pos_0!')
            else:
                # Refuel in pos_1
                pos = dp.loc[i+1, 'end_lat']
                dp.loc[i+1, 'predicted_price'] = min_prices_pred
                dp.loc[i+1, 'true_price'] = min_prices_true
                if min_ind[1] == 0:
                    dp.loc[i+1, 'refuel_time'] = arrival_pos_1
                else:
                    dp.loc[i+1, 'refuel_time'] = start_pos_1

                dp.loc[i+1, 'refuel_vol'] = car_fuel_vol - (dp.loc[i, 'fuel_vol_end'] + dp.loc[i+1, 'trip_con'])
                dp.loc[i+1, 'refuel_cost'] = dp.loc[i+1, 'refuel_vol'] * dp.loc[i+1, 'true_price']
                dp.loc[i, 'fuel_vol_start'] = car_fuel_vol
                if verbose:
                    print('Min price in pos_1!')
    return dp
