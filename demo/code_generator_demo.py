from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

import yaml

# System prompt — 专注于为设备做二次开发插件（强制要求先获取 SDK 文档 与 框架代码）
SYSTEM_PROMPT = """You are an expert microscopy systems engineer and developer who writes device integration plugins.
Workflow rules:
1) Before generating any plugin code, always require the user to provide:
   - the device SDK/driver documentation (API reference, examples), and
   - the host control software framework code (the plugin API / adapter interface).
2) If either SDK doc or framework code is missing, ask the user to upload them (or paste the relevant excerpts).
3) Once both are provided, generate a ready-to-deploy plugin module that fits the provided framework:
   - produce files and exact code snippets, with filenames and placement instructions.
   - include error handling, safety limits, connection/open/close semantics, and short usage example.
   - keep code concise and idiomatic Python, reference SDK calls exactly as shown in the provided SDK doc.
4) When producing code, include a short checklist of required runtime dependencies and tests to run before deploying.
5) Do not attempt to guess undocumented SDK behaviors; ask clarifying questions when needed.
"""

# Context schema
@dataclass
class Context:
    """Runtime context storing uploaded artifacts and operator info."""
    operator_id: str
    uploaded_sdk_docs: dict = field(default_factory=dict)      # name -> text
    uploaded_framework_files: dict = field(default_factory=dict)  # filename -> code
    metadata: dict = field(default_factory=dict)

# Tools to receive and inspect uploaded documents (they store content in context)
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
    This tool deliberately returns a scaffold text; full production code should be produced by the agent (LLM)
    once it confirms all required artifacts are available.
    """
    if not runtime.context.uploaded_sdk_docs or not runtime.context.uploaded_framework_files:
        return "Missing SDK doc or framework files. Please upload both before generating plugin."
    # simple scaffold (placeholder) — agent/LLM should later expand into full code using SDK details
    fn = target_filename or f"{plugin_name}.py"
    scaffold = (
        f"# {fn}\n"
        f"# Generated plugin scaffold for {plugin_name}\n\n"
        "class Plugin:\n"
        "    def __init__(self, connection_params):\n"
        "        self.connection_params = connection_params\n"
        "        self.connected = False\n\n"
        "    def connect(self):\n"
        "        # call SDK connect here\n"
        "        self.connected = True\n\n"
        "    def close(self):\n"
        "        # call SDK disconnect here\n"
        "        self.connected = False\n\n"
        "    def perform_action(self):\n"
        "        # implement action using SDK calls\n"
        "        pass\n"
    )
    runtime.context.metadata['last_generated'] = {"plugin": plugin_name, "filename": fn}
    return scaffold

# Load credentials/config (原来的 yaml 文件读取逻辑，保留以便后续模型配置)
try:
    with open("api_keys.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    cfg = {}

provider = "deepseek"
provider_cfg = cfg.get(provider, {}) if isinstance(cfg, dict) else {}
model_id = provider_cfg.get("model")
api_key = provider_cfg.get("api_key")

# Configure model (保持与原示例相同的初始化方式)
model = init_chat_model(
    model=model_id,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key
)

# Response format for this domain
@dataclass
class ResponseFormat:
    """Response schema for the plugin-generation agent."""
    assistant_message: str
    files: dict | None = None            # filename -> code
    actions: list[str] | None = None     # steps/checklist

# Memory / checkpointer
checkpointer = InMemorySaver()

# Create agent with tooling for uploading docs and generating plugin
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[upload_sdk_doc, upload_framework_file, inspect_artifacts, generate_plugin_stub],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# Example interaction flow (演示)
config = {"configurable": {"thread_id": "plugin-dev-1"}}

# Step 1: user asks to generate a plugin (agent should request SDK and framework)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我想为公司显微相机 X 编写一个二次开发插件，能帮我生成吗？"}]},
    config=config,
    context=Context(operator_id="dev_1")
)
print("STEP 1:", response['structured_response'])

# Step 2: user uploads SDK doc (simulated paste)
sdk_text = "CameraX SDK v1.2\nconnect(device_id): opens connection\ncapture(exposure_ms)->bytes: captures image\nclose(): closes connection\n"
response = agent.invoke(
    {"messages": [{"role": "user", "content": "这是设备 SDK 文档：\n---SDK_START---\n" + sdk_text + "\n---SDK_END---"}]},
    config=config,
    context=Context(operator_id="dev_1")
)
print("STEP 2:", response['structured_response'])

# Step 3: user uploads framework adapter interface (simulated)
framework_code = "class HostPluginInterface:\n    def initialize(self, params):\n        pass\n    def shutdown(self):\n        pass\n    def execute(self, command, **kwargs):\n        pass\n"
response = agent.invoke(
    {"messages": [{"role": "user", "content": "这是我的控制软件的 plugin 接口代码：\n---FRAMEWORK_START---\n" + framework_code + "\n---FRAMEWORK_END---"}]},
    config=config,
    context=Context(operator_id="dev_1")
)
print("STEP 3:", response['structured_response'])

# Step 4: user asks agent to produce the plugin now that docs are provided
response = agent.invoke(
    {"messages": [{"role": "user", "content": "现在请根据上面的 SDK 文档和框架接口，生成可以直接部署到我的框架下的插件代码，并给出文件名、依赖清单和部署步骤。"}]},
    config=config,
    context=Context(operator_id="dev_1")
)
print("STEP 4:", response['structured_response'])