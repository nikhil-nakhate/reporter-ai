"""
Utility functions for the Reporter AI application.
"""
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_error_response(message: str) -> str:
    """Create a JSON error response."""
    return json.dumps({"type": "error", "content": message})


def create_chunk_response(content: str) -> str:
    """Create a JSON chunk response."""
    return json.dumps({"type": "chunk", "content": content})


def create_done_response() -> str:
    """Create a JSON done response."""
    return json.dumps({"type": "done"})


def check_qwen_dependencies() -> Dict[str, bool]:
    """
    Check which Qwen dependencies are available.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    deps_status = {}
    dependencies = {
        "torch": lambda: __import__("torch"),
        "transformers": lambda: __import__("transformers"),
        "qwen_vl_utils": lambda: __import__("qwen_vl_utils"),
        "bitsandbytes": lambda: __import__("bitsandbytes"),
        "PIL": lambda: __import__("PIL")
    }
    
    for dep_name, import_func in dependencies.items():
        try:
            import_func()
            deps_status[dep_name] = True
        except ImportError:
            deps_status[dep_name] = False
    
    return deps_status

