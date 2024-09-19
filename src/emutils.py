import pandas as pd
import streamlit as st
import chromadb as ch
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction # type: ignore
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

OPEN_API_KEY = st.secrets["OPEN_API_KEY"]
ORG_ID = st.secrets["ORG_ID"]

if "openai_model_em" not in st.session_state:
    st.session_state["openai_model_em"] = "text-embedding-ada-002"

chroma_client = ch.HttpClient()


openai_ef = OpenAIEmbeddingFunction(
    api_key = OPEN_API_KEY,
    model_name = st.session_state["openai_model_em"]
)


# Test the embedding generation
test_input = "This is a test input."
try:
    embedding = openai_ef(test_input)
    print("Embedding successful:", embedding[:5])  # Print first few elements of the embedding
except Exception as e:
    print(f"Error generating embedding: {e}")

#Create corpus function (DF transform and attached conbined column)

def generate_corpus(df: pd.DataFrame, exclude: list):
    # Exclude the columns listed in 'exclude'
    # Generate combined text from the remaining columns
    combined = df.apply(lambda row: ', '.join(map(str, [row[col] for col in df.columns if col not in exclude])), axis=1)
    # Create a new DataFrame with the combined text
    df_combined = pd.DataFrame({'combined': combined})
    df = df.join(df_combined)
    return df

def clean_metadata(metadata_list):
    cleaned_metadata = []
    for metadata in metadata_list:
        cleaned_entry = {str(k): v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        cleaned_metadata.append(cleaned_entry)
    return cleaned_metadata

# Session state for collection creation checking
def is_collection_empty(col_name : str) -> bool : # type: ignore
    try:
        chroma_collection = chroma_client.get_collection(name = col_name)
        return False
    except Exception as e:
        print("Collection not found")
        return True

#Create (Store) in Vector DB (Chroma) with em function
def add_collection(df : pd.DataFrame, name : str):

    if 'prod_id' not in df.columns or 'combined' not in df.columns:
        raise ValueError("Missing required columns: 'prod_id' or 'combined'")
    
    # Check for null values in prod_id
    if df.prod_id.isnull().any():
        raise ValueError("Null values found in 'prod_id'")
    
    # Log the first few IDs, documents, and metadata for debugging
    print("Adding collection:", name)
    print("Sample IDs:", df.prod_id.astype(str).to_list()[:5])
    print("Sample Documents:", df.combined.to_list()[:5])
    print("Sample Metadata:", df.loc[:, df.columns != "combined"].to_dict('records')[:5])


    collection = chroma_client.get_or_create_collection(name = name, 
                                                        embedding_function = openai_ef,
                                                        # metadata={"hnsw:space": "ip"})
                                                        metadata={"hnsw:space": "cosine"})
    
    cleaned_metadatas = clean_metadata(df.loc[:, df.columns != "combined"].to_dict('records'))

    collection.add(
        ids = df.prod_id.astype(str).to_list(),
        documents = df.combined.to_list(),
        metadatas = cleaned_metadatas
    )

def get_collection(name : str):
    try:
        collection = chroma_client.get_collection(name = name)
    except Exception as e:
        print(f"Get collection error : {e}")
        
    return collection


def delete_collection(col_name : str):   
    
    try:
        chroma_client.delete_collection(name = col_name)
        print(f"Collection : {col_name} was sucessful delete.")
    except Exception as e:
        print(f"Delete was error : {e}")

def delete_all_collection():
    try:
        collections = chroma_client.list_collections()
        for collection in collections:
            collection_name = collection.name
            print(f"Deleting collection: {collection_name}")
            chroma_client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Delete was error : {e}")

def delete_doc(col_name : str, ids : list, condition : dict):
    collection = chroma_client.get_collection(name = col_name)
    try:
        collection.delete(ids=ids,where = condition)
        print("Delete data sucess")
    except Exception as e:
        print(f"Error from data delete : {e}")

def upsert_collection(col_name : str, df : pd.DataFrame):
    collection = chroma_client.get_collection(name = col_name)
    cleaned_metadatas = clean_metadata(df.loc[:, df.columns != "combined"].to_dict('records'))
    try:
        collection.upsert(
            ids = df.prod_id.astype(str).to_list(),
            documents = df.combined.to_list(),
            metadatas = cleaned_metadatas
        )
        print(f" Collection : {col_name} was sucessful upsert.")
    except Exception as e:
        print(f" Upsert error : {e}")

def visualize_embeddings(df, labels, title="Embedding Visualization"):
    # Convert embeddings to a 2D space using t-SNE
    document = df.combined.to_list()
    embeddings = openai_ef(document)  # Assuming openai_ef produces a list of embeddings
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(8, 8))
    for i, label in enumerate(labels):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.text(x+0.02, y+0.02, label, fontsize=9)
    
    plt.title(title)
    plt.show()

def check_similarity(query, documents):
    query_embedding = openai_ef([query])[0]  # Get the embedding for the query
    document_embeddings = openai_ef(documents)  # Get embeddings for documents

    similarities = cosine_similarity([query_embedding], document_embeddings)[0]  # type: ignore # Compare query to documents
    ranked_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # Print document similarities
    for idx, similarity in ranked_similarities:
        print(f"Document {idx} similarity: {similarity}, Document text: {documents[idx]}")

def calculate_embedding_distances(text: str, collection_name: str, max_result: int = 5):
    # Step 1: Get collection from ChromaDB
    collection = chroma_client.get_collection(name=collection_name)
    
    # Step 2: Generate the embedding for the query text
    query_embedding = openai_ef([text])[0]  # Generating a single embedding for the query
    
    # Step 3: Retrieve all document embeddings, documents, and metadata from ChromaDB
    try:
        # Retrieve embeddings, documents, and metadata (no 'ids' in the include list)
        all_documents = collection.get(include=["embeddings", "documents", "metadatas"])   # type: ignore
        document_embeddings = all_documents["embeddings"]  # List of document embeddings
        document_texts = all_documents["documents"]  # List of document texts
        document_metadatas = all_documents["metadatas"]  # List of metadata dictionaries
    except Exception as e:
        print(f"Error retrieving data from ChromaDB: {e}")
        return pd.DataFrame()  # Return empty DataFrame if there's an error

    # Step 4: Extract product IDs from the metadata (assuming 'prod_id' is stored in metadata)
    document_ids = [meta.get("prod_id", "unknown") for meta in document_metadatas]  # type: ignore # Default to "unknown" if missing

    # Step 5: Calculate cosine similarities between query embedding and document embeddings
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]  # type: ignore # 1D array of similarities
    
    # Step 6: Sort the results by similarity (highest first)
    ranked_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    # Step 7: Select the top results based on the max_result parameter
    top_results = ranked_results[:max_result]
    
    # Step 8: Prepare the results in a DataFrame format
    results = []
    for idx, similarity in top_results:
        result = {
            'prod_id': document_ids[idx],  # Extracted from metadata
            'similarity': similarity,
            'metadata': document_metadatas[idx], # type: ignore
            'document': document_texts[idx] # type: ignore
        }
        results.append(result)
    
    # Convert to DataFrame for easy display and further processing
    df_results = pd.DataFrame(results)
    # Check similarity score
    df_final_results, record_count = check_similarity_score(df_results)
    if record_count != 0:
        # Display the results
        print(f"Top {max_result} results for query: '{text}'")
        print(df_final_results)
    else:
        print(f"Relavant records is : {record_count}")

    return df_final_results, record_count

def check_similarity_score(df : pd.DataFrame):
    threshold = 0.8
    filter_df = df[df['similarity'] >= threshold]
    return filter_df, len(filter_df)



#Get Embedding value function (Debug purposed)

# def get_embedding(text, model):
#     # Get datafrom with 
#     text = text.replace("\n", " ")
#     # em = openai_client.embedding.create(input = [text], model = model).data[0].embedding
#     em = openai_ef(text)
#     return em

# def generate_vectorDF(df : pd.DataFrame, exclude : np.List):
#     # Combine data meta data to get corpus
#     df = generate_corpus(df, exclude)

#     df['text_embedded'] = df.combined.apply(lambda x : get_embedding(x , model = st.session_state["openai_model_em"]))
#     return df

# Collection query and return similar rank function

# def query_collection(text : str, col_name : str,  max_result : int) -> pd.DataFrame :
#     collection = chroma_client.get_collection(name = col_name)
#     text_em = openai_ef(text)
#     try:
#         query_result = collection.query(
#             # query_texts = text,
#             query_embeddings=text_em,
#             n_results = max_result,
#             include=["metadatas", "documents", "distances"])
#         print("Query success!")
#     except Exception as e:
#         print(f"Query error : {e}")
    
#     df = pd.DataFrame({'id':query_result['ids'][0],'score':query_result['distances'][0]}) # type: ignore
#     df_meta = pd.DataFrame.from_dict(query_result['metadatas'][0]) # type: ignore
#     df_result = pd.concat([df,df_meta], axis=1, join = "inner")

#     return df_result