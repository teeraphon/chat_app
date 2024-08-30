import streamlit as st
from src import ui

st.logo("./src/image/logo-5.png")
# ui.display_sidebar_header()
pg = st.navigation([st.Page("P1_Data_Showcase.py", title= "DATA SHOWCASE", icon=":material/database:"), 
                    st.Page("P2_Chatbot.py", title="CHATBOT", icon=":material/smart_toy:")])
pg.run()
