import pandas as pd
import numpy as np
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

rides = pd.read_csv('raw data/2017-fordgobike-tripdata.csv', \
                    parse_dates=['start_time', 'end_time'], infer_datetime_format=True)

# drop observations that start or end outside SF

rides = rides[rides['start_station_longitude'] < -122.35]
rides = rides[rides['end_station_longitude'] < -122.35]

# drop observations without gender
rides = rides[~rides['member_gender'].isnull()]

# drop observations that are longer than 2 hours
rides['hours'] = rides['duration_sec'] / 3600
rides = rides[rides['hours'] < 2]


# create variables of interest
# hour of day start, month, day of week, distance
def haversine(lat1, lon1, lat2, lon2):
    """ Returns distance between two points given lat and lon
    arguments: lat1 - latitude of first point type: float
    lon1 - longitude of first point type: float
    lat2 - latitude of second point type: float
    lon2 - longitude of second point type: float"""

    MILES = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    total_miles = MILES * c
    return total_miles


rides['distance'] = haversine(rides['start_station_latitude'],
                              rides['start_station_longitude'],
                              rides['end_station_latitude'],
                              rides['end_station_longitude'])

rides['mph'] = rides['distance'] / rides['hours']

rides['weekday'] = rides['start_time'].dt.weekday
rides['month'] = rides['start_time'].dt.month
rides['hour'] = rides['start_time'].dt.hour
rides['DATE'] = rides['start_time'].dt.date
rides['begin_time'] = rides['start_time'].dt.time
rides['stop_time'] = rides['end_time'].dt.time

# Read in weather data

weather = pd.read_csv('raw data/1335737.csv', parse_dates=['DATE'], infer_datetime_format=True, low_memory=False)

# keep only variables of interest: wind speed, precip, temp, sunrise, sunset
weather = weather[['DATE', 'HOURLYDRYBULBTEMPF', 'HOURLYPrecip',
                   'HOURLYWindSpeed', 'DAILYSunrise', 'DAILYSunset']]
weather['DATETIME'] = weather['DATE']
weather['DATE'] = weather['DATETIME'].dt.date
# get unique data for sunrise sunset and classify ride as before or after dark

sunrise_sunset = weather.drop_duplicates(subset='DATE')[['DATE', 'DAILYSunrise', 'DAILYSunset']]
sunrise_sunset['Sunrise'] = pd.to_datetime(sunrise_sunset['DAILYSunrise'], format='%H%M').dt.time
sunrise_sunset['Sunset'] = pd.to_datetime(sunrise_sunset['DAILYSunset'], format='%H%M').dt.time

rides_sunset = pd.merge(rides, sunrise_sunset, on="DATE", how='left')

condition = ((rides_sunset['begin_time'] < rides_sunset['Sunrise']) |
             (rides_sunset['begin_time'] > rides_sunset['Sunset']) |
             (rides_sunset['stop_time'] < rides_sunset['Sunrise']) |
             (rides_sunset['stop_time'] > rides_sunset['Sunset']))

rides_sunset['after_dark'] = np.where(condition, 1, 0)
rides_sunset['male_dummy'] = np.where(rides_sunset['member_gender'] == 'Male', 1, 0)

# clean up weather to get precipitation, temp and wind variables
weather['hour'] = weather['DATETIME'].dt.hour
weather['Rain'] = pd.to_numeric(weather['HOURLYPrecip'], errors='coerce')
weather['Temp'] = pd.to_numeric(weather['HOURLYDRYBULBTEMPF'], errors='coerce')
# group by hour and day and get average hourly temp, wind speed and precip

grouped = weather.groupby(['DATE', 'hour'], as_index=False)
weather_hourly = grouped[['Rain', 'HOURLYWindSpeed', 'Temp']].agg(np.mean)
weather_hourly = weather_hourly.fillna(method='ffill')
weather_hourly.info()

rides_weather = pd.merge(rides_sunset, weather_hourly, on=["DATE", 'hour'], how='left')
# get age as attribute
rides_weather['age'] = 2017 - rides_weather['member_birth_year']

# ages over 100 set to mean
rides_weather['age'] = np.where(rides_weather['age'] > 95, rides_weather['age'].mean(), rides_weather['age'])

# cluster stations
unique_stations = rides_weather.drop_duplicates(subset=['start_station_latitude', 'start_station_longitude'])[
    ['start_station_latitude', 'start_station_longitude']]
coords = unique_stations.as_matrix(columns=['start_station_latitude', 'start_station_longitude'])
kms_per_radian = 6371.0088
epsilon = 0.30 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine', algorithm='ball_tree')
db.fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


centermost_points = clusters.map(get_centermost_point)
lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'lon': lons, 'lat': lats})

rep_points.reset_index(inplace=True)

unique_stations.head()

# merge clusters back onto stations

unique_stations['clusters'] = cluster_labels
unique_stations = pd.merge(unique_stations, rep_points, left_on='clusters', right_on='index')
unique_stations.head()

# merge clusters and station centers onto start and stop stations
unique_stations = unique_stations.drop('index', axis=1)
unique_stations.rename(columns={'lat': 'start_cluster_latitude',
                                'lon': 'start_cluster_longitude',
                                'clusters': 'start_cluster'}, inplace=True)
unique_stations.head()

rides_weather = pd.merge(rides_weather, unique_stations, on=['start_station_latitude', 'start_station_longitude'])
rides_weather.head()

unique_stations.rename(columns={'start_cluster_latitude': 'end_cluster_latitude',
                                'start_cluster_longitude': 'end_cluster_longitude',
                                'start_cluster': 'end_cluster',
                                'start_station_latitude': 'end_station_latitude',
                                'start_station_longitude': 'end_station_longitude'}, inplace=True)

rides_weather = pd.merge(rides_weather, unique_stations, on=['end_station_latitude', 'end_station_longitude'])

rides_weather.head()

# Get dummies for weekday, hour, user type, month, cluster)

dummies = pd.get_dummies(rides_weather,
                         columns=['end_cluster', 'start_cluster', 'user_type', 'hour', 'month', 'weekday'])

# reset missing age to mean
dummies['age'] = np.where(dummies['age'].isnull(), dummies['age'].mean(), dummies['age'])
pickle.dump(dummies, open("cleaned data/rides.pkl", "wb"))

# output csv for tableau visualizations
dummies.to_csv("'cleaned data/rides.csv")
