
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    print("DuckDuckGoSearchRun available")
except ImportError:
    print("DuckDuckGoSearchRun NOT available")
