import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from src import ui

st.title("Let's Chat!")

product_path = './data/product.json'
order_path = './data/order.json'

product = pd.read_json(product_path)
order = pd.read_json(order_path)

product_df = pd.DataFrame(product)
order_df = pd.DataFrame(order)

OPEN_API_KEY = st.secrets["OPEN_API_KEY"]
ORG_ID = st.secrets["ORG_ID"]

openai_client = OpenAI(
    api_key = OPEN_API_KEY,
    organization = ORG_ID
)

if "openai_model_em" not in st.session_state:
    st.session_state["openai_model_em"] = "text-embedding-ada-002"

if "openai_model_lm" not in st.session_state:
    st.session_state["openai_model_lm"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the message with streamlit chat display widget
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
# Recieve and Handling user message input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role" : "user", "content" : prompt})
    #Disply message
    with st.chat_message("user"):
        st.markdown(prompt)
    #Prepare to disply assistance response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in openai_client.chat.completions.create(
            model = st.session_state["openai_model_lm"],
            messages = [{"role" : m["role"], "content" : m["content"]} for m in st.session_state.messages],
            stream = True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Appending the assistant's response to the session's message list
    st.session_state.messages.append({"role": "assistant", "content": full_response})
