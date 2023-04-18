# %% [markdown]
# Spatial modelling

# %%
import pandas as pd
import geopandas as gpd
from shapely import Point


aq_data = pd.read_csv("data/air_quality.csv")
geometry = [Point(xy) for xy in zip(aq_data["longitude"], aq_data["latitude"])]
aq_data = gpd.GeoDataFrame(aq_data, geometry=geometry)


# # Download air quality data directly from a given url
# df = pd.read_csv(
#     "https://data.london.gov.uk/download/breathe-london-aqmesh-pods/267507cc-9740-4ea7-be05-4d6ae16a5e4a/stationary_data.csv",
#     low_memory=False,
# )[["latitude", "longitude", "date_utc", "no2_ugm3"]]

# df = df.set_index("date_utc")
# df.index = pd.to_datetime(df.index)
# df = df.loc["2019-02-15 09:00"].reset_index(drop=True)
# # # Slice out 9am on 15th February 2019
# # hour_of_interest = dt.datetime(year=2019, month=2, day=15, hour=9)
# # spatial_data = slice_hour(aq_data, of_interest)
# # geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
# # gpd.GeoDataFrame(df, geometry=geometry).to_file("data/air_quality.shp")
# # spatial_data.to_file("datasets/aq/aq.shp")

# %%
