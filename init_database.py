import os
import sys
import argparse
import time
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "OpenMCS_Agent"))

from OpenMCS_Agent.utils.document_loader import load_and_split_documents
from OpenMCS_Agent.tools.rag_tool import ensure_vector_store
from OpenMCS_Agent.core.schemas import Context
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

class MockRuntime:
    def __init__(self):
        self.context = Context(operator_id="init_script")

def compute_file_hash(params):
    # Just a dummy or real hash if we had file access, 
    # but docs already loaded. We can hash content.
    return hashlib.md5(params.encode('utf-8')).hexdigest()

def step1_load_and_split(folder_path):
    print("--- Step 1: Loading and Splitting Documents ---")
    documents = load_and_split_documents(folder_path)
    print(f"Total documents chunks loaded: {len(documents)}")
    return documents

def step2_enhance_metadata(documents):
    print("--- Step 2: Enhancing Metadata ---")
    enhanced_docs = []
    for doc in documents:
        # Ensure page_content is a string
        if doc.page_content is None:
            doc.page_content = ""
        elif not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)

        # Ensure cleanup of content (remove null bytes)
        doc.page_content = doc.page_content.replace('\x00', '')

        # Skip empty documents (many embedding APIs fail on empty strings)
        if not doc.page_content.strip():
            # optional: print skipped file
            # print(f"Skipping empty document from {doc.metadata.get('source', 'unknown')}")
            continue

        # Add basic metadata if missing
        if "source" not in doc.metadata:
            doc.metadata["source"] = "unknown"
        
        # Add timestamp
        doc.metadata["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add content hash for deduplication/tracking
        content_hash = compute_file_hash(doc.page_content)
        doc.metadata["content_hash"] = content_hash
        enhanced_docs.append(doc)
            
    
    # Filter out complex metadata (lists/dicts) which ChromaDB cannot handle
    enhanced_docs = filter_complex_metadata(enhanced_docs)
    
    print(f"Enhanced metadata for {len(enhanced_docs)} valid documents (skipped {len(documents) - len(enhanced_docs)} empty/invalid).")
    return enhanced_docs

def step3_vectorize_and_store(documents):
    print("--- Step 3: Vectorization and Storage ---")
    if not documents:
        print("No documents to store.")
        return

    runtime = MockRuntime()
    vs = ensure_vector_store(runtime.context)
    
    # Try smaller batch size
    batch_size = 10
    total = len(documents)
    
    print(f"Vectorizing and storing {total} chunks into database...")
    
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        try:
            vs.add_documents(batch)
            # print(f"Stored batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\nBatch starting at {i} failed: {e}")
            print("Retrying documents one by one to isolate error...")
            success_count = 0
            for doc in batch:
                try:
                    vs.add_documents([doc])
                    success_count += 1
                except Exception as inner_e:
                    print(f"Failed to add document from {doc.metadata.get('source', 'unknown')}")
                    print(f"Content preview: {doc.page_content[:50]}...")
                    print(f"Error: {inner_e}")
            print(f"Recovered {success_count}/{len(batch)} documents from failed batch.")
            
    if hasattr(vs, "persist"):
        vs.persist()
    print("\nStorage complete.")

def init_database(folder_path):
    # Health Check
    print("--- Health Check: Embedding Service ---")
    try:
        from OpenMCS_Agent.tools.rag_tool import get_embeddings
        emb = get_embeddings()
        res = emb.embed_query("test string for health check")
        print(f"Embedding service is working. Vector dimension: {len(res)}")
    except Exception as e:
        print(f"CRITICAL ERROR: Embedding service check failed. Please check your api_keys.yaml or network connection.\nError details: {e}")
        return

    # Step 1
    docs = step1_load_and_split(folder_path)
    
    if not docs:
        print("No documents found to process.")
        return
        
    # Step 2
    docs = step2_enhance_metadata(docs)
    
    # Step 3
    step3_vectorize_and_store(docs)
    
    print("\nDatabase initialization finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Vector Database from files.")
    parser.add_argument("folder", nargs="?", help="Path to the source folder containing documents", default=None)
    
    args = parser.parse_args()
    
    source_folder = args.folder
    if not source_folder:
        source_folder = input("Please enter the source folder path: ").strip()
        # For testing/default behavior if no arg provided
        # source_folder = os.path.join(current_dir, "resources")
        # print(f"No folder provided, using default: {source_folder}")
        
    if os.path.exists(source_folder):
        init_database(source_folder)
    else:
        print(f"Error: Folder '{source_folder}' does not exist.")
