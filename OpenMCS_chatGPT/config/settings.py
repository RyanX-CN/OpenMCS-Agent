import yaml
import os

langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
if langchain_endpoint.startswith("<") and langchain_endpoint.endswith(">"):
    langchain_endpoint = langchain_endpoint[1:-1]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint

def load_config(config_path="api_keys.yaml"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        # 尝试在上级目录查找
        config_path = os.path.join("..", config_path)
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: {config_path} not found.")
        return {}

def get_available_models():
    """获取配置文件中所有可用的配置名称（最外层键名）"""
    cfg = load_config()
    if not isinstance(cfg, dict):
        return []
    # 直接返回最外层的 key，例如 ['DeepSeek-Chat', 'OpenAI-GPT4']
    return list(cfg.keys())

def get_model_config(config_name=None):
    """根据配置名称获取详细配置"""
    cfg = load_config()
    
    # 如果没有指定，默认使用第一个
    if not config_name:
        if cfg:
            config_name = list(cfg.keys())[0]
        else:
            return {}

    # 获取对应的配置块
    model_cfg = cfg.get(config_name, {})
    
    return {
        "model_id": model_cfg.get("model_id", ""),
        "provider": model_cfg.get("provider", ""),
        "api_key": model_cfg.get("api_key", ""),
        "config_name": config_name
    }