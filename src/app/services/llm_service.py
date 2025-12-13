"""
Service for LLM text generation with streaming support.
Supports multiple LLM providers: Anthropic, OpenAI, Qwen VLM, etc.
"""
import os
import json
import logging
from typing import AsyncIterator, Optional, Dict, List

from anthropic import Anthropic
from dotenv import load_dotenv

from app.config import AVAILABLE_LLMS
from app.utils import create_error_response, create_chunk_response, create_done_response

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Try to import Qwen adapter
try:
    from app.services.qwen_adapter import QwenAdapter, QWEN_AVAILABLE
except ImportError:
    QwenAdapter = None
    QWEN_AVAILABLE = False


class LLMService:
    """Service for LLM text generation with support for multiple providers."""
    
    def __init__(self):
        """Initialize the LLM service (no specific model, will be selected per request)."""
        self.anthropic_client: Optional[Anthropic] = None
        self.qwen_adapters: Dict[str, QwenAdapter] = {}
        self._init_clients()
    
    def _init_clients(self):
        """Initialize API clients for different providers."""
        # Initialize Anthropic client if key is available
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.anthropic_client = Anthropic(api_key=anthropic_key)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Initialize Qwen adapters if available (lightweight - doesn't load models)
        if QWEN_AVAILABLE and QwenAdapter:
            for model_id, model_info in AVAILABLE_LLMS.get("qwen", {}).items():
                try:
                    qwen_model_id = model_info.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
                    adapter = QwenAdapter(model_id=qwen_model_id)
                    # Lightweight availability check (doesn't load model)
                    if adapter.is_available():
                        self.qwen_adapters[model_id] = adapter
                        logger.info(f"✅ Qwen adapter created for {model_id} (model will load on first use)")
                    else:
                        logger.warning(f"❌ Qwen adapter for {model_id} not available")
                except Exception as e:
                    logger.warning(f"❌ Failed to create Qwen adapter for {model_id}: {e}")
                    logger.debug(f"Qwen adapter error details: {e}", exc_info=True)
        else:
            if not QWEN_AVAILABLE:
                logger.info("ℹ️  QwenVLM not available - dependencies may be missing")
            elif not QwenAdapter:
                logger.info("ℹ️  QwenAdapter not available")
        
        # OpenAI client can be initialized here when needed
        # For now, we'll initialize it lazily
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get list of available LLM models.
        
        Returns:
            List of model dictionaries with id, name, provider, and availability
        """
        models = []
        for provider, provider_models in AVAILABLE_LLMS.items():
            for model_id, model_info in provider_models.items():
                # Check availability based on provider
                if provider == "qwen":
                    # For Qwen, check if adapter is available
                    available = model_id in self.qwen_adapters
                else:
                    # For API-based models, check if API key is available
                    requires_key = model_info.get("requires_key")
                    if requires_key:
                        api_key = os.environ.get(requires_key)
                        available = api_key is not None and api_key.strip() != ""
                    else:
                        available = True
                
                models.append({
                    "id": model_id,
                    "name": model_info["name"],
                    "provider": model_info["provider"],
                    "available": available
                })
        return models
    
    async def stream_generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming support.
        
        Args:
            model_id: Model identifier (e.g., "claude-3-5-sonnet")
            system_prompt: System prompt for the model
            user_prompt: User prompt/content
            max_tokens: Maximum tokens to generate
            
        Yields:
            Text chunks as they are generated
        """
        # Find the model configuration
        model_config = None
        for provider_models in AVAILABLE_LLMS.values():
            if model_id in provider_models:
                model_config = provider_models[model_id]
                break
        
        if not model_config:
            yield create_error_response(f"Model '{model_id}' not found")
            return
        
        provider = model_config["provider"]
        
        # Route to appropriate provider
        if provider == "anthropic":
            async for chunk in self._stream_anthropic(model_id, system_prompt, user_prompt, max_tokens):
                yield chunk
        elif provider == "openai":
            async for chunk in self._stream_openai(model_id, system_prompt, user_prompt, max_tokens):
                yield chunk
        elif provider == "qwen":
            async for chunk in self._stream_qwen(model_id, system_prompt, user_prompt, max_tokens):
                yield chunk
        else:
            yield create_error_response(f"Provider '{provider}' not yet implemented")
    
    async def _stream_anthropic(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream generation using Anthropic API."""
        if not self.anthropic_client:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_key:
                yield create_error_response("ANTHROPIC_API_KEY not set")
                return
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        
        try:
            with self.anthropic_client.messages.stream(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield create_chunk_response(text)
                
                yield create_done_response()
                
        except Exception as e:
            logger.error(f"Error in Anthropic stream generation: {e}", exc_info=True)
            yield create_error_response(str(e))
    
    async def _stream_openai(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream generation using OpenAI API."""
        try:
            from openai import OpenAI
            
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                yield create_error_response("OPENAI_API_KEY not set")
                return
            
            client = OpenAI(api_key=openai_key)
            
            stream = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield create_chunk_response(chunk.choices[0].delta.content)
            
            yield create_done_response()
            
        except ImportError:
            yield create_error_response("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Error in OpenAI stream generation: {e}", exc_info=True)
            yield create_error_response(str(e))
    
    async def _stream_qwen(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int
    ) -> AsyncIterator[str]:
        """Stream generation using Qwen VLM."""
        if model_id not in self.qwen_adapters:
            # Try to create adapter on the fly
            try:
                model_config = AVAILABLE_LLMS.get("qwen", {}).get(model_id, {})
                qwen_model_id = model_config.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
                adapter = QwenAdapter(model_id=qwen_model_id)
                if adapter.is_available():
                    self.qwen_adapters[model_id] = adapter
                else:
                    yield create_error_response("QwenVLM model not available or failed to load")
                    return
            except Exception as e:
                logger.error(f"Failed to initialize Qwen adapter: {e}")
                yield create_error_response(f"Failed to load Qwen model: {str(e)}")
                return
        
        adapter = self.qwen_adapters[model_id]
        async for chunk in adapter.stream_generate(system_prompt, user_prompt, max_tokens):
            yield chunk

