import yaml
import os

langsmith_api = os.getenv("LANGCHAIN_API_KEY", "")
if langsmith_api.startswith("<") and langsmith_api.endswith(">"):
    langsmith_api = langsmith_api[1:-1]

# LangSmith configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api
os.environ["LANGCHAIN_PROJECT"] = "OpenMCS-Agent"

def load_config(config_path="api_keys.yaml"):
    """Load config from a YAML file."""
    if not os.path.exists(config_path):
        config_path = os.path.join("..", config_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: {config_path} not found.")
        return {}

def get_available_models():
    """Get list of available chat model configuration names."""
    cfg = load_config()
    if not isinstance(cfg, dict):
        return []
    chat_models = cfg.get("Available chat model", {})
    return list(chat_models.keys())

def get_model_config(config_name=None):
    """Get detailed configuration based on the model name"""
    cfg = load_config()
    
    chat_models = cfg.get("Available chat model", {})

    if not config_name:
        if chat_models:
            config_name = list(chat_models.keys())[0]
        else:
            return {}

    model_cfg = chat_models.get(config_name, {})
    
    return {
        "model_id": model_cfg.get("model_id", config_name),
        "provider": model_cfg.get("provider", ""),
        "api_key": model_cfg.get("api_key", ""),
        "base_url": model_cfg.get("url", ""),
        "config_name": config_name
    }

def get_embedding_config(config_name=None):
    """Get embedding configuration"""
    cfg = load_config()
    
    embedding_models = cfg.get("Available embedding model", {})

    if not config_name:
        if embedding_models:
            config_name = list(embedding_models.keys())[0]
        else:
            return {}

    model_cfg = embedding_models.get(config_name, {})
    
    return {
        "model_id": model_cfg.get("model_id", config_name),
        "provider": model_cfg.get("provider", ""),
        "api_key": model_cfg.get("api_key", ""),
        "base_url": model_cfg.get("url", ""),
        "config_name": config_name
    }