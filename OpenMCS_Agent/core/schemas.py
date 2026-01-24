from dataclasses import dataclass, field
from typing import Any

@dataclass
class Context:
    """运行时上下文，存储上传的文档和操作员信息"""
    operator_id: str
    uploaded_sdk_docs: dict = field(default_factory=dict)      # name -> text
    uploaded_framework_files: dict = field(default_factory=dict)  # filename -> code
    metadata: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)                 # key -> value (Long-term memory)
    vector_store: Any = None                                   # VectorStore instance (Persistent)
    temp_vector_store: Any = None                              # VectorStore instance (Temporary)

@dataclass
class ResponseFormat:
    """Agent 的结构化响应格式"""
    assistant_message: str
    files: dict | None = None            # filename -> code
    actions: list[str] | None = None     # steps/checklist