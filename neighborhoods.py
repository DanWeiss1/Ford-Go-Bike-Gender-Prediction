import pandas as pd
import geopandas
import pickle
from shapely.geometry import Point, Polygon

"""
Gets the neighborhoods of each station from SF shapefile
"""

rides = pd.read_pickle('cleaned data/rides.pkl')
# get list of unique_stations
unique_stations = rides.drop_duplicates(subset=['start_station_latitude', 'start_station_longitude'])[
    ['start_station_latitude', 'start_station_longitude', 'start_station_name']]

# read in data on neighborhood boundaries

sf_hoods = geopandas.read_file('raw data/geo_export_048bf835-497f-406a-ab03-d6233d5d8ec9.shp')
# convert data to dict for ease of analysis
sf_hoods.set_index('name', inplace=True)
sf_hoods_dict = sf_hoods['geometry'].to_dict()
# make geoseries with points for each station
pnts = unique_stations.apply(lambda row: Point(row['start_station_longitude'],
                                               row['start_station_latitude']), axis=1)

# make dataframe with point geography, neighborhood
pnts2 = geopandas.GeoDataFrame(geometry=pnts.values, index=pnts.index)
pnts3 = pd.merge(pnts2, unique_stations, left_index=True, right_index=True)
pnts3.set_index(['start_station_name'], inplace=True)
pnts3.drop(['start_station_latitude', 'start_station_longitude'], axis=1, inplace=True)
pnts4 = pnts3.assign(**{key: pnts3.within(geometry) for key, geometry in sf_hoods_dict.items()})
pnts4.drop('geometry', inplace=True, axis=1)
pnts5 = pnts4.astype(int)
pnts5 = pnts5.loc[:, (pnts5 != 0).any(axis=0)]
start_cols = ['start_' + col for col in pnts5.columns]
end_cols = ['end_' + col for col in pnts5.columns]

# merge neighborhoods back onto rides
pnts5.columns = start_cols
pnts5.head()
rides2 = pd.merge(rides, pnts5, left_on='start_station_name', right_index=True)
pnts5.columns = end_cols
pnts5.head()
rides3 = pd.merge(rides2, pnts5, left_on='end_station_name', right_index=True)
rides3.head()

pickle.dump(rides3, open("cleaned data/rides_hoods.pkl", "wb"))
