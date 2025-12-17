from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from pprint import pprint

def load_html(path: str):
    loader = UnstructuredHTMLLoader(path, mode="elements", strategy="fast")
    docs = loader.load()
    print(f"Loaded {len(docs)} HTML documents from {path}")
    return docs

def load_pdf(path: str):
    loader = PyPDFLoader(path, mode="page")
    docs = loader.load()
    print(f"Loaded {len(docs)} PDF documents from {path}")
    return docs

def load_json(path: str):
    loader = JSONLoader(path, jq_schema=".")
    docs = loader.load()
    print(f"Loaded {len(docs)} JSON documents from {path}")
    return docs

def load_source_code(directory: str):
    loader = GenericLoader.from_filesystem(
        directory,
        glob="*",
        suffixes=[".py", ".cpp", ".c", ".h", ".java"],
        parser=LanguageParser(),
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} source code documents from {directory}")
    return docs

if __name__ == "__main__":
    file_path = "./resource/pdf_file/SCAS0134E_C13440-20CU_tec.pdf"
    docs = load_pdf(file_path)
    print(len(docs))
    for document in docs:
        pprint(document.metadata)
    # print("\n\n--> o <--\n\n".join([document.page_content for document in data]))