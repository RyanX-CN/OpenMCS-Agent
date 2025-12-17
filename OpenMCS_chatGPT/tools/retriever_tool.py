from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool



def retriever_tool(documents):
    """Retrieve top-k similar documents for a given query."""
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents, embedding=OpenAIEmbeddings(
            base_url="https://ai.nengyongai.cn/v1",
            api_key="sk-t5dQSnFuC6zinhgNc7eKu1Gx2e5STeRF4DoDI9uCo4wB6KD0",
            model = "text-embedding-3-large")
    )
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "Document Retriever",
        "Useful for retrieving relevant documents based on a query.",
    )
    return retriever_tool


if __name__ == "__main__":
    from document_loader import load_pdf
    docs = load_pdf("./resource/pdf_file/SCAS0134E_C13440-20CU_tec.pdf")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50)
    docs_splits = text_splitter.split_documents(docs)
    docs_splits[0].page_content.strip()
    tool = retriever_tool(docs_splits)
    print("===================== Retriever Tool Test ====================")
    tool.batch([
        "which image formats are supported?",
        "what's the max frame rate?",],)