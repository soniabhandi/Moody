from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model = 'sentence-transformers/all-MiniLM-L6-v2'
)

texts = [
    "hi how are you",
    "good morning world",
    "go back to your place"
]

vector1 = embeddings.embed_documents(texts)
vector = embeddings.embed_query("You are going to create RAG")
print(vector)

