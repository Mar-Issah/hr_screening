from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4)),
]
db = FAISS.from_documents(list_of_documents, embeddings)

# results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=1))
# Or with a callable:
results_with_scores = db.similarity_search_with_score("foo", filter=lambda d: d["page"] == 1)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")