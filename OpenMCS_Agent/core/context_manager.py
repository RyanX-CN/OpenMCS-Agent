import threading
from typing import Optional
from core.schemas import Context

# Use a global variable instead of threading.local to support tools running in thread pools
# This assumes a single active agent session at a time, which is true for this desktop app.
_global_context: Optional[Context] = None

def set_active_context(context: Context):
    global _global_context
    _global_context = context

def get_active_context() -> Optional[Context]:
    return _global_context
