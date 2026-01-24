from langchain.tools import tool
from core.schemas import Context
from core.context_manager import get_active_context

@tool
def save_memory(key: str, value: str) -> str:
    """Save a piece of information to long-term memory."""
    context = get_active_context()
    if not context: return "Error: Context not active."
    context.memory[key] = value
    return f"Memory saved: {key}"

@tool
def read_memory(key: str) -> str:
    """Read a piece of information from long-term memory."""
    context = get_active_context()
    if not context: return "Error: Context not active."
    return context.memory.get(key, "Memory not found.")

@tool
def list_memories() -> str:
    """List all keys in long-term memory."""
    context = get_active_context()
    if not context: return "Error: Context not active."
    return str(list(context.memory.keys()))
