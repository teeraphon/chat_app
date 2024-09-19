from re import split
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from src import emutils

import requests
from requests.exceptions import ConnectionError

        
st.title("Let's Chat!")

product_path = './data/product.json'
order_path = './data/order.json'

product = pd.read_json(product_path)
order = pd.read_json(order_path)

product_df = pd.DataFrame(product)
order_df = pd.DataFrame(order)

product_list = product_df['prod'].unique()
product_list = list(map(lambda x : x.lower(), product_list))
order_list = order_df['prod'].unique()
order_list = list(map(lambda x : x.lower(), order_list))

OPEN_API_KEY = st.secrets["OPEN_API_KEY"]
ORG_ID = st.secrets["ORG_ID"]

openai_client = OpenAI(
    api_key = OPEN_API_KEY,
    organization = ORG_ID
)

if "openai_model_lm" not in st.session_state:
    st.session_state["openai_model_lm"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "messages_prompt" not in st.session_state:
    st.session_state.messages_prompt = []

#Init product collection
if emutils.is_collection_empty(col_name = "product_collection"):
    # Dataframe processing, create corpus data (Combination of row meta data)
    try:
        product_corpus = emutils.generate_corpus(product_df,['prod_id'])
        print("Create corpus for Product sucess!")
    except Exception as e:
        print(f"Create corpus error : {e}")

    try:
        emutils.add_collection(product_corpus, "product_collection")
        print("Add collection to chroma DB success!")
    except Exception as e:
        print(f"Add collection to Chroma DB error : {e}")
    
# Init order collection
if emutils.is_collection_empty(col_name = "order_collection"):
    # Dataframe processing, create corpus data (Combination of row meta data)
    try:
        order_corpus = emutils.generate_corpus(order_df,['prod_id'])
        print("Create corpus for Order sucess!")
    except Exception as e:
        print(f"Create corpus error : {e}")

    try:
        emutils.add_collection(order_corpus, "order_collection")
        print("Add order collection to chroma DB success!")
    except Exception as e:
        print(f"Add order collection to Chroma DB error : {e}")

# Search product collection
def check_related_product(text : str):
    col_name = "product_collection"
    print("==Result ranking from PRODUCT COLLECTION==")
    result_df,count = emutils.calculate_embedding_distances(text=text, collection_name=col_name, max_result=3)
    return result_df,count

# Search order collection
def check_related_order(text : str):
    col_name = "order_collection"
    print("==Result ranking from ORDER COLLECTION==")
    result_df,count = emutils.calculate_embedding_distances(text=text, collection_name=col_name, max_result=3)
    return result_df,count

# Show the message with streamlit chat display widget
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

# Recieve and Handling user message input
if prompt := st.chat_input("What's up?"):

    st.session_state.messages.append({"role" : "user", "content" : prompt})
    st.session_state.messages_prompt.append({"role" : "user", "content" : prompt})

    result_prod , count_prod = check_related_product(prompt)
    result_order, count_order = check_related_order(prompt)

    # Check for short question is consist of product
    is_short = bool(len(prompt.split()) <= 2)
    product_s = any(prod_element in prompt.lower() for prod_element in product_list)
    order_s =  any(order_element in prompt.lower() for order_element in order_list)

    # Init prompt to OpenAI
    st.session_state.messages_prompt.append({"role" : "system", "content" : """You are a highly knowledgeable assistant helping customers with beauty-related product recommendations, order history, and store suggestions. 
                                                You should be friendly, conversational, and respond in detail, providing context for recommendations 
                                                and addressing user questions thoroughly.When users inquire about products or orders, 
                                                reference their purchase history and provide relevant advice. Ensure your responses are polite 
                                                and engaging."""})
    st.session_state.messages_prompt.append({"role" : "system", "content" : """Ensure that all responsed is related to beauty product."""})
    st.session_state.messages_prompt.append({"role" : "system", "content" : """
                                                If the user greets you or asks casual questions (like 'How are you?'), respond politely and warmly. 
                                                If the user makes a general inquiry, ask follow-up questions to understand their needs better."""})
    st.session_state.messages_prompt.append({"role" : "system", "content" : """Ensure that all responses are detailed, providing comprehensive 
                                                explanations and suggestions. Avoid brief answers unless explicitly requested."""})

    if(count_order != 0 or bool(is_short and order_s)):

        prev_purchases = ". ".join([f"{row['document']}" for index, row in result_order.iterrows()]) # type: ignore
        st.session_state.messages_prompt.append({"role" : "user", "content" : f"Here're my latest product orders: {prev_purchases}"})
                
    else:
        st.session_state.messages_prompt.append({"role" : "assistant", "content" : "There are no related product from my purchase history"})

    # Found some related product
    if(count_prod != 0 or bool(is_short and product_s)):
        products_list = []
        st.session_state.messages_prompt.append({"role" : "user", "content" : "Please give me a detailed explanation of your recommendations"})
        st.session_state.messages_prompt.append({"role" : "user", "content" : "Please be friendly and talk to me like a person, don't just give me a list of recommendations"})
        st.session_state.messages_prompt.append({"role" : "assistant", "content" : f"""I found {count_prod} related products for you. 
                                                 I'll give you a personalized recommendation and explain why each product might be 
                                                 a good fit for your needs based on your preferences. These are list of related product:"""})
        
        for index, row in result_prod.iterrows(): # type: ignore
            product_dict = {"role" : "assistant", "content" : f"Brand : {row["metadata"]["brand"]}. Product description : {row["document"]}"}
            products_list.append(product_dict)

        st.session_state.messages_prompt.extend(products_list)
        st.session_state.messages_prompt.append({"role" : "assistant", "content" : "Here's my summarized recommendation of products, and why it would suit you:"})

    # No related product found
    else:
        st.session_state.messages_prompt.append({"role" : "system", "content" : """Ensure that responses are gentle infrom customer that there are no product matching they needs right now. You need to recommend a litle bit on their search."""})
        st.session_state.messages_prompt.append({"role": "assistant", "content": "I'm sorry, I couldn't find a matching beauty product in our store. Could you tell me more about your preferences?"})
    
    if count_prod == 0 or count_order == 0:
        st.session_state.messages_prompt.append({"role" : "assistant", "content" : """It looks like there are no matching products or orders 
                                                    at the moment. Could you tell me more about what you're looking for? 
                                                    I'll do my best to help you find the right product!"""})
 
    # Handling location seperately
    if "store" in prompt.lower() or "shop" in prompt.lower() or "location" in prompt.lower():
        try:
            if len(st.session_state['near_store']) != 0:
                try:
                    current_address = st.session_state['current_address']
                    print(f"My current address is {current_address}")
                except Exception as e:
                    print(f"Curreent address is not init")
                st.session_state.messages_prompt.append({"role" : "user", f"content" : "This is my current location : {current_address}. Please give me nearby store."})
                st.session_state.messages_prompt.append({"role": "assistant", "content": f"""I found {len(st.session_state['near_store'])} shopping mall or our stores nearby your current location. 
                                                            I will give you a details for transportation to arrived at our store. Here are list and details of nearby store:"""})
                store_list = []
                for index, row in st.session_state['near_store'].iterrows():
                    store_info = f"{row['name']}, {row['distance']} km away, approximately {row['time']} mins by car."
                    store_list.append({"role": "assistant", "content": store_info})
                print(store_list)
                st.session_state.messages_prompt.extend(store_list)
            else:
                st.session_state.messages_prompt.append({"role": "assistant", "content": "It seems you're far from our partner stores. Would you like to order online?"})
        except Exception as e:
            print(f"Nearest store session is not init. : {e}")

    #Disply message
    with st.chat_message("user"):
        st.markdown(prompt)
    #Prepare to disply assistance response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in openai_client.chat.completions.create(
            model = st.session_state["openai_model_lm"],
            messages = [{"role" : m["role"], "content" : m["content"]} for m in st.session_state.messages_prompt],
            stream = True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Appending the assistant's response to the session's message list
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.messages_prompt.append({"role": "assistant", "content": full_response})
