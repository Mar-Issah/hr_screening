from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
# from langchain.llms import HuggingFaceHub
from langchain_pinecone import PineconeVectorStore
import os
import time
from langchain_chroma import Chroma

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME")


# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# iterate over files uploaded
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

def push_to_pinecone(docs, embeddings):
    # Pinecone(api_key=pinecone_api_key)
    index_name = pinecone_index_name
    index = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    return index

def push_to_chromadb(docs, embeddings):
    db = Chroma.from_documents(docs, embeddings)
    return db

#Function to help us get relavant documents from vector store - based on user input
def similarity_search(docsearch,query,k,unique_id):
     # similarity_search_with_score returns score % assign to each seacrh doc
    time.sleep(20)
    matched_docs = docsearch.similarity_search_with_score(query, int(k), {"unique_id":unique_id})
    return matched_docs


# Summarise doc
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary
