import datetime as dt

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def slice_hour(df: pd.DataFrame, timestamp):
    """
    Extract an hour slice of a spatiotemporal dataset.
    """
    df = df[["latitude", "longitude", "date_utc", "no2_ugm3"]]
    df["timestamp"] = pd.to_datetime(df.date_utc)
    return df[df["timestamp"] == timestamp].dropna().drop(["timestamp", "date_utc"], axis=1).reset_index(drop=True)


if __name__ == "__main__":
    # Download air quality data directly from a given url
    aq_data = pd.read_csv(
        "https://data.london.gov.uk/download/breathe-london-aqmesh-pods/267507cc-9740-4ea7-be05-4d6ae16a5e4a/stationary_data.csv"
    )

    # Slice out 9am on 15th February 2019
    hour_of_interest = dt.datetime(year=2019, month=2, day=15, hour=9)
    spatial_data = slice_hour(aq_data, of_interest)

    spatial_data.to_file("datasets/aq/aq.shp")
