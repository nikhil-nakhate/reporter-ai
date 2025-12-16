"""
Configuration constants for the Reporter AI application.
"""
# Available LLM models configuration
AVAILABLE_LLMS = {
    "anthropic": {
        "claude-3-5-sonnet": {
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "requires_key": "ANTHROPIC_API_KEY"
        },
        "claude-3-5-sonnet-20240620": {
            "name": "Claude 3.5 Sonnet (20240620)",
            "provider": "anthropic",
            "requires_key": "ANTHROPIC_API_KEY"
        },
        "claude-3-opus-20240229": {
            "name": "Claude 3 Opus",
            "provider": "anthropic",
            "requires_key": "ANTHROPIC_API_KEY"
        },
        "claude-3-sonnet-20240229": {
            "name": "Claude 3 Sonnet",
            "provider": "anthropic",
            "requires_key": "ANTHROPIC_API_KEY"
        },
        "claude-3-haiku-20240307": {
            "name": "Claude 3 Haiku",
            "provider": "anthropic",
            "requires_key": "ANTHROPIC_API_KEY"
        }
    },
    "openai": {
        "gpt-4-turbo-preview": {
            "name": "GPT-4 Turbo",
            "provider": "openai",
            "requires_key": "OPENAI_API_KEY"
        },
        "gpt-4": {
            "name": "GPT-4",
            "provider": "openai",
            "requires_key": "OPENAI_API_KEY"
        },
        "gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "provider": "openai",
            "requires_key": "OPENAI_API_KEY"
        }
    },
    "qwen": {
        "qwen-2.5-vl-7b-instruct": {
            "name": "Qwen 2.5 VL 7B Instruct",
            "provider": "qwen",
            "requires_key": None,  # Local model, no API key needed
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"
        }
    }
}

# Default model
DEFAULT_LLM_MODEL = "claude-3-5-sonnet"

# Streaming chunk size for Qwen (characters per chunk)
QWEN_STREAM_CHUNK_SIZE = 10
QWEN_STREAM_DELAY = 0.01  # seconds

# TTS Model Configuration
# Available models:
# - "tts_models/multilingual/multi-dataset/xtts_v2" (default, recommended)
# - "tts_models/multilingual/multi-dataset/xtts_v1.1" (alternative, may have better transformers compatibility)
# - "tts_models/multilingual/multi-dataset/your_tts" (alternative)
# - "voice_conversion_models/multilingual/vctk/freevc24" (voice conversion model)
# Note: openvoice_v2 is NOT available in TTS library - it's a separate standalone library from myshell-ai
# If you want standalone OpenVoice, you need to install it separately (see OPENVOICE_OPTIONS.md)
DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/your_tts"  # Works with transformers>=4.46.2!

