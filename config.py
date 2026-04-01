import os
from dotenv import load_dotenv


# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# from neo4j import GraphDatabase
# from pinecone import Pinecone

load_dotenv()

WEATHER_API = os.getenv("WEATHER_API")

#pinecone
# PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index("youtube-rag")


#Neo4j
# NEO4J_URI      = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


#nvidia
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL        = "https://integrate.api.nvidia.com/v1"
NVIDIA_THINK_MODEL = "moonshotai/kimi-k2-instruct-0905"  # 0.83s, best query quality
NVIDIA_GEN_MODEL   = "moonshotai/kimi-k2-instruct"

#groq
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL          = "https://api.groq.com/openai/v1"
THINK_MODEL            = "llama-3.3-70b-versatile"
THINK_MODEL_ALT        = "moonshotai/kimi-k2-instruct-0905"
GENERATION_MODEL       = "meta/llama-4-scout-17b-16e-instruct"
GENERATION_MODEL_ALT   = "openai/gpt-oss-120b"

#embedding_model
# embeddings = NVIDIAEmbeddings(
#     model="baai/bge-m3",
#     api_key=NVIDIA_API_KEY,
#     base_url="https://integrate.api.nvidia.com/v1"
# )

#Neo4j driver
# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

#telegram
API_ID   = os.getenv('TELEGRAM_API_ID')          # ← from my.telegram.org
API_HASH = (os.getenv('TELEGRAM_API_HASH'))   # ← from my.telegram.org
PHONE    = os.getenv('TELEGRAM_PHONE_NUMBER')   # ← your phone number