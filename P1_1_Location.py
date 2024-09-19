import numpy as np
import pandas as pd
import streamlit as st

import requests
from requests.exceptions import ConnectionError

import googlemaps
from datetime import datetime

#Direct fetch shop data
store_path = './data/sephora_store.json'
store = pd.read_json(store_path)
store_df = pd.DataFrame(store)

st.title("Store & User location")

key = st.secrets["GOOGLE_API_DIST"]

gmaps = googlemaps.Client(key=key)

def geocoding_address(address : str) -> dict:
    # Geocoding an address
    geocode_result = gmaps.geocode(address) # type: ignore
    loc_code = geocode_result[0]["geometry"]["location"]
    return loc_code


def geocoding_la_lon(location : tuple) -> str: # type: ignore
    # Look up an address with reverse geocoding
    reverse_geocode_result = gmaps.reverse_geocode(location) # type: ignore
    address_approx = reverse_geocode_result[0]["formatted_address"]
    return address_approx

def get_distance_matrix(origin : str, destination : str):
    now = datetime.now()
    directions_matrix = gmaps.distance_matrix(origin, # type: ignore
                                     destination,
                                     mode="driving",
                                     departure_time=now)
    raw_time = directions_matrix["rows"][0]["elements"][0]["duration"]["value"]
    raw_distance = directions_matrix["rows"][0]["elements"][0]["distance"]["value"]

    time_min, time_sec = divmod(raw_time, 60)
    distance = raw_distance / 1000

    return {'time' : time_min,'distance' : distance}

# Function to get geolocation data from Google API
def get_geolocation():

    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={key}'
    data = {
        "considerIp": "true"  # Uses the IP address of the request to get location
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching geolocation data")
        return None

location_data = get_geolocation()

if "current_loc" not in st.session_state:
    st.session_state.current_loc = location_data

if location_data:
    current_lat = location_data['location']['lat']
    current_lng = location_data['location']['lng']
    accuracy = location_data['accuracy']
    current_location = (current_lat,current_lng)
    current_address = geocoding_la_lon(current_location)

    if "current_address" not in st.session_state:
        st.session_state.current_address = current_address

    # Display location data
    st.success(f"""LATITUDE: {current_lat}, LONGITUDE: {current_lng} \n
                   ADDRESS : {current_address}""")
    
    # Create a DataFrame to plot on the map
    current_loc_df = pd.DataFrame({'lat': [current_lat], 'lng': [current_lng],'size': 100, 'color': '#8530c2' })

# Apply geocoding function to get latitude and longitude
store_loc_coor = store_df['address'].apply(lambda x: geocoding_address(x))

# Convert the dictionary of lat/lng into a DataFrame
df_lat_lng = pd.DataFrame(store_loc_coor.tolist(), index=store_df.index)  # convert list of dicts to DataFrame

# Ensure that loc_meta contains information for each store (adjust as needed)
loc_meta = pd.DataFrame([{'size': 100, 'color': '#ffaa0088'}] * len(store_df), index=store_df.index)

# Join the latitude/longitude DataFrame to the original DataFrame using the index
store_df = store_df.join(df_lat_lng)

# Join loc_meta data
store_df = store_df.join(loc_meta)

map_df = pd.concat([store_df,current_loc_df])
st.map(map_df, latitude="lat", longitude="lng", size="size", color="color", zoom = 5)

st.divider()
st.subheader("Nearest store")

time_distance  = store_df['address'].apply(lambda x: get_distance_matrix(current_address,x))
df_time_dist = pd.DataFrame(time_distance.tolist(), index=store_df.index)
store_df = store_df.join(df_time_dist)

store_df = store_df.sort_values(by = ['time'])
store_nearest = store_df.copy()
store_nearest = store_nearest[(store_nearest['time'] <= 20) | (store_nearest['distance'] <= 10)]
for index , row in store_nearest.iterrows():
    st.markdown(body = f""" - {row['name']} : {row['distance']} km. far from current location and take {row['time']} mins approximately by driving.""")

with st.expander("All sorted store data from current loc."):
    st.write(store_df[['name','city','time','distance','address']])

if "near_store" not in st.session_state:
    st.session_state.near_store = store_nearest