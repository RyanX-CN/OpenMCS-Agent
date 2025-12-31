from langchain.tools import tool, ToolRuntime
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from core.schemas import Context
import os

# Configuration for Embeddings (You might want to move this to settings)
EMBEDDING_BASE_URL = "https://ai.nengyongai.cn/v1"
EMBEDDING_API_KEY = "sk-t5dQSnFuC6zinhgNc7eKu1Gx2e5STeRF4DoDI9uCo4wB6KD0" # Consider moving to env var
EMBEDDING_MODEL = "text-embedding-3-large"

def get_embeddings():
    return OpenAIEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
        model=EMBEDDING_MODEL
    )

def ensure_vector_store(context: Context):
    if context.vector_store is None:
        context.vector_store = InMemoryVectorStore(get_embeddings())
    return context.vector_store

@tool
def search_knowledge_base(runtime: ToolRuntime[Context], query: str) -> str:
    """Search the knowledge base (RAG) for relevant information."""
    vs = ensure_vector_store(runtime.context)
    # Retrieve top 3 results
    results = vs.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in knowledge base."
    
    return "\n\n".join([f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}" for doc in results])

@tool
def add_to_knowledge_base(runtime: ToolRuntime[Context], content: str, source: str = "user_input") -> str:
    """Add text content to the knowledge base."""
    vs = ensure_vector_store(runtime.context)
    doc = Document(page_content=content, metadata={"source": source})
    vs.add_documents([doc])
    return f"Added content to knowledge base (source: {source})."

