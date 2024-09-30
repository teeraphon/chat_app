import streamlit as st
from src import ui
from src import maputils
import pandas as pd

store_path = './data/sephora_store.json'
store = pd.read_json(store_path)
store_df = pd.DataFrame(store)

if "current_loc" not in st.session_state:
    location_data = maputils.get_geolocation()
    st.session_state.current_loc = location_data
    maputils.store_current_address()

if "near_store" not in st.session_state:
    store_nearest = maputils.get_near_store(store_df)
    st.session_state.near_store = store_nearest

st.logo("./src/image/logo-5.png")
# ui.display_sidebar_header()
pg = st.navigation([st.Page("P1_Data_Showcase.py", title= "DATA SHOWCASE", icon=":material/database:"),
                    st.Page("P1_1_Location.py", title= "LOCATION", icon=":material/location_on:"),  
                    st.Page("P2_Chatbot.py", title="CHATBOT", icon=":material/smart_toy:")])
pg.run()
