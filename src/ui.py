import os
from pathlib import Path
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from typing import Iterable
from typing import List
from typing import Text

def display_sidebar_header() -> None:
    logo = Image.open("./src/image/logo-1.png")
    with st.sidebar:
        st.image(logo, use_column_width=True)
        # col1 ,col2 = st.columns(2)
        # st.header("")