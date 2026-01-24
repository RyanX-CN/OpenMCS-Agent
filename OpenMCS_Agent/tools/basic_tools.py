from langchain.tools import tool
from langchain_core.documents import Document
from core.schemas import Context
from core.context_manager import get_active_context
from tools.rag_tool import ensure_vector_store

@tool
def upload_sdk_doc(name: str, content: str) -> str:
    """Store SDK/driver documentation text under a name."""
    context = get_active_context()
    if not context: return "Error: Context not active."
    context.uploaded_sdk_docs[name] = content
    
    # Also add to RAG
    vs = ensure_vector_store(context)
    doc = Document(page_content=content, metadata={"source": name, "type": "sdk_doc"})
    vs.add_documents([doc])
    
    return f"SDK document '{name}' uploaded ({len(content)} chars) and indexed in knowledge base."

@tool
def upload_framework_file(filename: str, content: str) -> str:
    """Store framework/plugin-interface file content."""
    context = get_active_context()
    if not context: return "Error: Context not active."
    context.uploaded_framework_files[filename] = content
    return f"Framework file '{filename}' uploaded ({len(content)} chars)."

@tool
def inspect_artifacts() -> str:
    """Return a short summary of what has been uploaded."""
    context = get_active_context()
    if not context: return "Error: Context not active."
    sdk_keys = list(context.uploaded_sdk_docs.keys())
    fw_keys = list(context.uploaded_framework_files.keys())
    return f"SDK docs: {sdk_keys}; framework files: {fw_keys}"

@tool
def generate_plugin_stub(plugin_name: str, target_filename: str = None) -> str:
    """
    A helper tool to produce a small plugin scaffold using the uploaded artifacts.
    """
    context = get_active_context()
    if not context: return "Error: Context not active."
    if not context.uploaded_sdk_docs or not context.uploaded_framework_files:
        return "Missing SDK doc or framework files. Please upload both before generating plugin."
    
    fn = target_filename or f"{plugin_name}.py"
    scaffold = (
        f"# {fn}\n# Generated plugin scaffold for {plugin_name}\n"
        "class Plugin:\n    pass\n"
    )
    context.metadata['last_generated'] = {"plugin": plugin_name, "filename": fn}
    return scaffold