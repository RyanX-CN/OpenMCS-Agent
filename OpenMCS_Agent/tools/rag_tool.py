from langchain.tools import tool, ToolRuntime
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.schemas import Context
from config.settings import get_embedding_config, get_model_config
import os, json, hashlib

try:
    from langchain_chroma import Chroma
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

def get_embeddings():
    cfg = get_embedding_config()
    return OpenAIEmbeddings(
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
        model=cfg.get("model_id"),
        check_embedding_ctx_length=False
    )

def get_chat_model():
    cfg = get_model_config()
    kwargs = {
        "model": cfg.get("model_id"),
        "temperature": 0,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 2,
        "api_key": cfg.get("api_key"),
        "model_provider": cfg.get("provider"),
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg.get("base_url")
    return init_chat_model(**kwargs)

def _project_root():
    # OpenMCS_chatGPT/tools -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _vector_store_dir():
    return os.path.join(_project_root(), "database")

def _manifest_path():
    return os.path.join(_vector_store_dir(), "manifest.json")

def _load_manifest():
    try:
        with open(_manifest_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}

def _save_manifest(manifest: dict):
    os.makedirs(_vector_store_dir(), exist_ok=True)
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

def ensure_vector_store(context: Context):
    if context.vector_store is not None:
        return context.vector_store

    os.makedirs(_vector_store_dir(), exist_ok=True)
    if HAS_CHROMA:
        context.vector_store = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=_vector_store_dir(),
            collection_name="openmcs_knowledge",
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        # Fallback to in-memory if Chroma is unavailable
        context.vector_store = InMemoryVectorStore(get_embeddings())
    return context.vector_store

@tool
def search_knowledge_base(runtime: ToolRuntime[Context], query: str) -> str:
    """Search the knowledge base (RAG) for relevant information."""
    vs = ensure_vector_store(runtime.context)
    # Retrieve top 4 results with relevance (if supported)
    try:
        results = vs.similarity_search_with_relevance_scores(query, k=4)
        if not results:
            return "No relevant information found in knowledge base."
        lines = []
        for doc, score in results:
            src = doc.metadata.get("source", doc.metadata.get("source_path", "unknown"))
            lines.append(f"Score: {score:.3f}\nSource: {src}\nContent: {doc.page_content}")
        return "\n\n".join(lines)
    except Exception:
        results = vs.similarity_search(query, k=4)
        if not results:
            return "No relevant information found in knowledge base."
        return "\n\n".join([f"Source: {doc.metadata.get('source', doc.metadata.get('source_path', 'unknown'))}\nContent: {doc.page_content}" for doc in results])

@tool
def add_to_knowledge_base(runtime: ToolRuntime[Context], content: str, source: str = "user_input") -> str:
    """Add text content to the knowledge base."""
    vs = ensure_vector_store(runtime.context)
    doc = Document(page_content=content, metadata={"source": source})
    try:
        vs.add_documents([doc])
        # Persist if using Chroma
        if HAS_CHROMA:
            vs.persist()
    except Exception:
        vs.add_documents([doc])
    return f"Added content to knowledge base (source: {source})."

def _make_chunk_docs(text: str, base_meta: dict, id_prefix: str):
    chunks = _splitter.split_text(text)
    docs = []
    ids = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        meta = dict(base_meta)
        meta["chunk_index"] = idx
        doc_id = f"{id_prefix}:{idx}"
        meta["doc_id"] = doc_id
        docs.append(Document(page_content=chunk, metadata=meta))
        ids.append(doc_id)
    return docs, ids

def _load_text_from_path(path: str) -> str:
    # Minimal loader: read as UTF-8 text; for binary, attempt ignore errors
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

@tool
def update_knowledge_base_from_files(runtime: ToolRuntime[Context], file_paths: str) -> str:
    """Index or update files into the persistent knowledge base. Accepts comma/semicolon-separated paths (files or directories)."""
    vs = ensure_vector_store(runtime.context)
    os.makedirs(_vector_store_dir(), exist_ok=True)
    manifest = _load_manifest()
    files_map = manifest.get("files", {})

    def _process_file(path: str):
        abspath = os.path.abspath(path)
        if not os.path.exists(abspath):
            return f"Skip (not found): {path}"
        # Compute file signature
        sig = _file_hash(abspath)
        mtime = os.path.getmtime(abspath)
        prev = files_map.get(abspath)
        # If exists and unchanged, skip
        if prev and prev.get("hash") == sig:
            return f"Up-to-date: {path}"
        # If exists and changed, delete previous docs
        if prev and prev.get("doc_ids"):
            try:
                if HAS_CHROMA and hasattr(vs, "delete"):
                    vs.delete(ids=prev["doc_ids"])  # type: ignore
            except Exception:
                pass
        # Load text and chunk
        base_meta = {"source_path": abspath, "source_mtime": mtime, "source_hash": sig}
        text = _load_text_from_path(abspath)
        if not text or not text.strip():
            return f"Skipped (empty file): {path}"

        id_prefix = f"{sig}"
        docs, ids = _make_chunk_docs(text, base_meta, id_prefix)
        
        if not docs:
            return f"Skipped (no valid chunks): {path}"

        # Add to store with batching
        BATCH_SIZE = 10
        try:
            total_docs = len(docs)
            for i in range(0, total_docs, BATCH_SIZE):
                batch_docs = docs[i : i + BATCH_SIZE]
                batch_ids = ids[i : i + BATCH_SIZE] if ids else None
                
                # Sanity check: Ensure pure strings for page_content
                for d in batch_docs:
                    if not isinstance(d.page_content, str):
                        d.page_content = str(d.page_content)
                    # Filter out null bytes or problematic chars if needed
                    d.page_content = d.page_content.replace('\x00', '')

                if HAS_CHROMA:
                    vs.add_documents(batch_docs, ids=batch_ids)  # type: ignore
                else:
                    vs.add_documents(batch_docs)
            
            if HAS_CHROMA:
                vs.persist()
                    
        except Exception as e:
            # If embedding fails, log it and return error for this file but don't crash
            return f"Error indexing {path}: {str(e)}"
        
        # Update manifest
        # Update manifest
        files_map[abspath] = {"hash": sig, "mtime": mtime, "doc_ids": ids}
        return f"Indexed: {path} ({len(ids)} chunks)"

    outputs = []
    # Split incoming paths
    parts = [p.strip() for p in file_paths.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if os.path.isdir(p):
            # Index all files in directory (text-based)
            for root, _, fnames in os.walk(p):
                for fname in fnames:
                    fpath = os.path.join(root, fname)
                    outputs.append(_process_file(fpath))
        else:
            outputs.append(_process_file(p))

    manifest["files"] = files_map
    _save_manifest(manifest)
    return "\n".join(outputs)

@tool
def rag_answer(runtime: ToolRuntime[Context], question: str) -> str:
    """RAG pipeline: retrieve, score relevance; if low, rewrite question and retry; if high, extract info, rewrite question, and answer."""
    vs = ensure_vector_store(runtime.context)
    model = get_chat_model()
    # Retrieve with scores if possible
    try:
        results = vs.similarity_search_with_relevance_scores(question, k=5)
    except Exception:
        docs = vs.similarity_search(question, k=5)
        results = [(d, 0.0) for d in docs]

    if not results:
        # No docs: ask model to answer or ask for clarification
        prompt = (
            "你是专业的OpenMCS助手。当前知识库为空或未命中。\n"
            f"原始问题：{question}\n"
            "请在不臆造细节的前提下，改写更明确的问题，并给出可行的后续信息收集建议。"
        )
        resp = model.invoke(prompt)
        return str(getattr(resp, "content", resp))

    # Decide relevance by top score
    top_doc, top_score = results[0]
    relevance_threshold = 0.35  # heuristic; adjust as needed
    if top_score < relevance_threshold:
        # Rewrite question to improve retrieval
        prompt = (
            "你是专业的OpenMCS助手。当前检索结果相关性较低。\n"
            f"原始问题：{question}\n"
            "请在不增加虚假信息的情况下，基于领域常识改写更清晰的检索式（只输出改写后的问题）。"
        )
        rew = model.invoke(prompt)
        new_q = str(getattr(rew, "content", rew)).strip()
        # Retry once
        try:
            results = vs.similarity_search_with_relevance_scores(new_q, k=5)
        except Exception:
            docs = vs.similarity_search(new_q, k=5)
            results = [(d, 0.0) for d in docs]

    # Compose context from top documents
    selected = results[:4]
    context_blocks = []
    for doc, score in selected:
        src = doc.metadata.get("source", doc.metadata.get("source_path", "unknown"))
        context_blocks.append(f"[score={score:.3f}] 来源: {src}\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    # Ask model to extract key info and answer
    prompt = (
        "你是专业的OpenMCS助手。请从以下检索到的内容中提取与问题直接相关的关键信息，\n"
        "并基于这些信息对原问题进行更精确的改写，然后给出最终答案。\n\n"
        f"原始问题：{question}\n\n"
        f"检索上下文：\n{context_text}\n\n"
        "输出格式：\n"
        "1) 改写问题：<你的改写>\n"
        "2) 关键信息：<要点整理，引用必要来源>\n"
        "3) 答案：<最终答案>\n"
    )
    resp = model.invoke(prompt)
    return str(getattr(resp, "content", resp))

