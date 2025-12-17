from langchain.tools import tool, ToolRuntime
from core.schemas import Context

@tool
def upload_sdk_doc(runtime: ToolRuntime[Context], name: str, content: str) -> str:
    """Store SDK/driver documentation text under a name."""
    runtime.context.uploaded_sdk_docs[name] = content
    return f"SDK document '{name}' uploaded ({len(content)} chars)."

@tool
def upload_framework_file(runtime: ToolRuntime[Context], filename: str, content: str) -> str:
    """Store framework/plugin-interface file content."""
    runtime.context.uploaded_framework_files[filename] = content
    return f"Framework file '{filename}' uploaded ({len(content)} chars)."

@tool
def inspect_artifacts(runtime: ToolRuntime[Context]) -> str:
    """Return a short summary of what has been uploaded."""
    sdk_keys = list(runtime.context.uploaded_sdk_docs.keys())
    fw_keys = list(runtime.context.uploaded_framework_files.keys())
    return f"SDK docs: {sdk_keys}; framework files: {fw_keys}"

@tool
def generate_plugin_stub(runtime: ToolRuntime[Context], plugin_name: str, target_filename: str = None) -> str:
    """
    A helper tool to produce a small plugin scaffold using the uploaded artifacts.
    """
    if not runtime.context.uploaded_sdk_docs or not runtime.context.uploaded_framework_files:
        return "Missing SDK doc or framework files. Please upload both before generating plugin."
    
    fn = target_filename or f"{plugin_name}.py"
    scaffold = (
        f"# {fn}\n# Generated plugin scaffold for {plugin_name}\n"
        "class Plugin:\n    pass\n"
    )
    runtime.context.metadata['last_generated'] = {"plugin": plugin_name, "filename": fn}
    return scaffold