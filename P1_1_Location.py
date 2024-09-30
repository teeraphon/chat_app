import numpy as np
import pandas as pd
import streamlit as st
from src import maputils
from requests.exceptions import ConnectionError

#Direct fetch shop data
store_path = './data/sephora_store.json'
store = pd.read_json(store_path)
store_df = pd.DataFrame(store)

st.title("Store & User location")

if st.session_state["current_loc"]:
    current_lat = st.session_state["current_loc"]['location']['lat']
    current_lng = st.session_state["current_loc"]['location']['lng']
    accuracy = st.session_state["current_loc"]['accuracy']
    current_location = (current_lat,current_lng)
    current_address = maputils.geocoding_la_lon(current_location)

    # Display location data
    st.success(f"""LATITUDE: {current_lat}, LONGITUDE: {current_lng} \n
                   ADDRESS : {current_address}""")
    
    # Create a DataFrame to plot on the map
    current_loc_df = pd.DataFrame({'lat': [current_lat], 'lng': [current_lng],'size': 100, 'color': '#8530c2' })
else:
    st.error("Not found current location store in session state!")

store_df = maputils.convert_store_master(store_df)

map_df = pd.concat([store_df,current_loc_df])
st.map(map_df, latitude="lat", longitude="lng", size="size", color="color", zoom = 5)

st.divider()
st.subheader("Nearest store")

for index , row in st.session_state['near_store'].iterrows():
    st.markdown(body = f""" - {row['name']} : {row['distance']} km. far from current location and take {row['time']} mins approximately by driving.""")

with st.expander("All sorted store data from current loc."):
    all_store = maputils.get_all_dist_store(store_df)
    st.write(all_store[['name','city','time','distance','address']])