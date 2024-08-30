import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from src import ui

product_path = './data/product.json'
order_path = './data/order.json'

product = pd.read_json(product_path)
order = pd.read_json(order_path)

product_df = pd.DataFrame(product)
order_df = pd.DataFrame(order)

st.title("DATASET SHOWCASE")

st.subheader("Data I : Product Data")
st.table(product_df)

st.subheader("Data II : Purchase History Data")
st.table(order_df)

