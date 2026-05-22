from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  # 384 dimensions


# Ollama
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama2-7b-embedding-q4_0")


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# # single text
# text = "This is a sample text to be embedded."
# embedding = embeddings.embed_query(text)
# print(f"Embedding for single text: {len(embedding)}")


# embeds = embeddings.embed_documents(
#     ["This is the first documents.", "This is the second document/"]
# )

# print(f"Embeddings for multiple texts: {embeds}")
# print(f"Number of embeddings returned: {len(embeds)}")  # Should print 2
# print(
#     f"Length of each embedding: {len(embeds[0])}"
# )  # Should print 1536 for text-embedding-3-small
