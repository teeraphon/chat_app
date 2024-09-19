import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from src import ui

product_path = './data/product.json'
order_path = './data/order.json'
store_path = './data/sephora_store.json'

product = pd.read_json(product_path)
order = pd.read_json(order_path)
store = pd.read_json(store_path)

product_df = pd.DataFrame(product)
order_df = pd.DataFrame(order)
store_df = pd.DataFrame(store)

st.title("DATASET SHOWCASE")

st.subheader("Data I : Product Data")
st.table(product_df)

st.subheader("Data II : Purchase History Data")
st.table(order_df)

st.subheader("Data III : Partner Store Data (Sephora)")
st.table(store_df)

