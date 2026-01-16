from langchain_community.document_loaders import UnstructuredHTMLLoader, PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader, TextLoader, BSHTMLLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import os

def load_html(path: str):
    # Use BSHTMLLoader to extract all text, then split it. 
    # UnstructuredHTMLLoader with mode="elements" creates too many small distinct docs.
    try:
        loader = BSHTMLLoader(path, open_encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        print(f"BSHTMLLoader failed for {path}: {e}, trying UnstructuredHTMLLoader")
        loader = UnstructuredHTMLLoader(path, mode="single", strategy="fast")
        docs = loader.load()
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} HTML chunks from {path}")
    return docs

def load_pdf(path: str):
    loader = PyPDFLoader(path) # default mode is 'page' which chunks by page
    # Split pages further if needed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)
    print(f"Loaded {len(docs)} PDF chunks from {path}")
    return docs

def load_json(path: str):
    loader = JSONLoader(path, jq_schema=".")
    docs = loader.load()
    print(f"Loaded {len(docs)} JSON documents from {path}")
    return docs

def load_markdown(path: str):
    try:
        # Try structured load first
        loader = UnstructuredMarkdownLoader(path, mode="elements", strategy="fast")
        docs = loader.load()
    except Exception as e:
        print(f"UnstructuredMarkdownLoader failed for {path} (maybe missing dependencies), falling back to TextLoader. Error: {e}")
        loader = TextLoader(path, encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = loader.load_and_split(text_splitter=splitter)
    print(f"Loaded {len(docs)} Markdown chunks from {path}")
    return docs

def load_text_file(path: str):
    loader = TextLoader(path, encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)
    print(f"Loaded {len(docs)} Text chunks from {path}")
    return docs

def load_code_file(path: str, language: Language):
    # GenericLoader works on directories, so we target the file explicitly via glob
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    
    loader = GenericLoader.from_filesystem(
        directory,
        glob=filename,
        suffixes=[os.path.splitext(filename)[1]],
        parser=LanguageParser(language=language, parser_threshold=500)
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} code units from {path}")
    return docs

def load_and_split_documents(folder_path: str):
    """
    Recursively load and split documents from a folder based on extension.
    """
    all_docs = []
    abs_root = os.path.abspath(folder_path)
    
    if not os.path.exists(abs_root):
        print(f"Path does not exist: {abs_root}")
        return []

    print(f"Scanning {abs_root} for documents...")
    
    for root, dirs, files in os.walk(abs_root):
        # Modify dirs in-place to skip specific directories
        # Skip directories named 'bin', 'lib', or 'resources'
        dirs[:] = [d for d in dirs if d.lower() not in ['log', 'bin', 'lib', 'resources']]

        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            try:
                docs = []
                if ext == ".html" or ext == ".htm":
                    docs = load_html(file_path)
                elif ext == ".pdf":
                    docs = load_pdf(file_path)
                elif ext == ".json":
                    docs = load_json(file_path)
                elif ext == ".md":
                    docs = load_markdown(file_path)
                elif ext == ".py":
                    docs = load_code_file(file_path, Language.PYTHON)
                elif ext in [".cpp", ".c", ".h", ".hpp"]:
                    docs = load_code_file(file_path, Language.CPP)
                elif ext == ".java":
                    docs = load_code_file(file_path, Language.JAVA)
                elif ext == ".js" or ext == ".ts":
                    docs = load_code_file(file_path, Language.JS)
                # Skip unknown files to avoid binary garbage
                
                if docs:
                    # Enforce metadata
                    for doc in docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["filename"] = file
                        doc.metadata["extension"] = ext
                    all_docs.extend(docs)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return all_docs
