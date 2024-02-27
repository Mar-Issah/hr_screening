from langchain.vectorstores import Pinecone
# from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
# import pinecone
from pypdf import PdfReader
# # from langchain.llms.openai import OpenAI
# from langchain.chains.summarize import load_summarize_chain
# from langchain.llms import HuggingFaceHub
from langchain_pinecone import PineconeVectorStore
import os
import streamlit as st
# from pinecone import Pinecone
# import time

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME")


# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        chunks=get_pdf_text(filename)
        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))
    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#old
def push_to_pinecone(docs, embeddings):
    # Pinecone(api_key=pinecone_api_key)
    index_name = pinecone_index_name
    index = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    return index


#Function to help us get relavant documents from vector store - based on user input
def mmr_search(docsearch,query,unique_id):
     # similarity_search_with_score returns with score % assign to each seacrh doc
    k = 1
    matched_docs = docsearch.similarity_search_with_score(query, int(k), {"unique_id":unique_id.strip()})
    print(unique_id)
    return matched_docs


#Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(pinecone_index_name,embeddings):
   # Pinecone has eliminated .init method
    Pinecone(api_key=pinecone_api_key)
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    return index


#Function to help us get relavant documents from vector store - based on user input
def similar_docs_pinecone(query, k, embeddings, unique_id):
    # Pinecone has eliminated .init method
    Pinecone(api_key=pinecone_api_key)
    index = pull_from_pinecone(pinecone_index_name,embeddings)
    # similarity_search_with_score returns with score % assign to each seacrh doc
    similar_docs = index.similarity_search_with_score(query, int(k), {"unique_id":unique_id})
    # similar_docs = index.similarity_search(query, filter = {"unique_id":unique_id})
    return similar_docs






#-------------------------------------------------------------------------------

#Function to pull infrmation from Vector Store - Pinecone here
# def pull_from_pinecone(pinecone_index_name,embeddings):
#     index_name = pinecone_index_name
#     index = PineconeVectorStore.from_existing_index(index_name, embeddings)
#     return index



# Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,embeddings,unique_id):
    index = pinecone_index_name
    index = pull_from_pinecone(index,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


# # Helps us get the summary of a document
# def get_summary(current_doc):
#     llm = OpenAI(temperature=0)
#     #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
#     chain = load_summarize_chain(llm, chain_type="map_reduce")
#     summary = chain.run([current_doc])
#     return summary
