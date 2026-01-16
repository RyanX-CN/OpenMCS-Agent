from langchain.tools import tool, ToolRuntime
from core.schemas import Context

@tool
def save_memory(runtime: ToolRuntime[Context], key: str, value: str) -> str:
    """Save a piece of information to long-term memory."""
    runtime.context.memory[key] = value
    return f"Memory saved: {key}"

@tool
def read_memory(runtime: ToolRuntime[Context], key: str) -> str:
    """Read a piece of information from long-term memory."""
    return runtime.context.memory.get(key, "Memory not found.")

@tool
def list_memories(runtime: ToolRuntime[Context]) -> str:
    """List all keys in long-term memory."""
    return str(list(runtime.context.memory.keys()))
