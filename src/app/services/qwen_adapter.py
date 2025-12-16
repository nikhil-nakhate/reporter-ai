"""
Adapter for Qwen VLM to work with the LLM service.
Uses local QwenVLM implementation.
"""
import json
import logging
import asyncio
from typing import AsyncIterator, Optional

from app.config import QWEN_STREAM_CHUNK_SIZE, QWEN_STREAM_DELAY
from app.utils import (
    create_error_response,
    create_chunk_response,
    create_done_response,
    check_qwen_dependencies
)

logger = logging.getLogger(__name__)

# Try to import local QwenVLM and check dependencies
QWEN_AVAILABLE = False
QwenVLM = None  # type: ignore
import_error = None

try:
    # First check if basic dependencies are available
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    
    # If we get here, dependencies are available, try importing QwenVLM
    from app.services.qwen_vlm import QwenVLM
    QWEN_AVAILABLE = True
    logger.info("✅ QwenVLM imported successfully from local implementation")
except (ImportError, SystemError) as e:
    # SystemError can occur when TensorFlow fails to initialize (e.g., numpy version mismatch)
    import_error = e
    logger.warning(f"QwenVLM not available: {e}")
    
    # Check which specific dependencies are missing
    deps_status = check_qwen_dependencies()
    missing_deps = [dep for dep, available in deps_status.items() if not available]
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
    QwenVLM = None  # type: ignore
    QWEN_AVAILABLE = False


class QwenAdapter:
    """Adapter to use QwenVLM with streaming support."""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize Qwen adapter.
        
        Args:
            model_id: Qwen model identifier
        """
        if not QWEN_AVAILABLE:
            raise ValueError("QwenVLM is not available. Make sure required dependencies are installed: torch, transformers, qwen_vl_utils, bitsandbytes, accelerate")
        
        self.model_id = model_id
        self.qwen_vlm: Optional[QwenVLM] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of QwenVLM."""
        if not self._initialized:
            try:
                logger.info(f"Initializing QwenVLM with model: {self.model_id}")
                self.qwen_vlm = QwenVLM(model_id=self.model_id)
                if self.qwen_vlm.model is None:
                    raise ValueError("Failed to load QwenVLM model")
                self._initialized = True
                logger.info("QwenVLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize QwenVLM: {e}")
                raise
    
    async def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming support using QwenVLM.
        
        Since QwenVLM doesn't have native streaming, we generate the full response
        and then stream it in chunks to provide a streaming-like experience.
        
        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt/content
            max_tokens: Maximum tokens to generate (passed to model)
            
        Yields:
            Text chunks as they are generated
        """
        self._ensure_initialized()
        
        if not self.qwen_vlm or not self.qwen_vlm.model:
            yield create_error_response("QwenVLM model not loaded")
            return
        
        try:
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            
            # Generate response in executor to avoid blocking
            # Use max_new_tokens from the parameter
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.qwen_vlm.generate_text_response(
                    prompt=full_prompt,
                    system_prompt=None,  # Already combined above
                    max_new_tokens=max_tokens
                )
            )
            
            # Stream the response in chunks to simulate streaming
            for i in range(0, len(response), QWEN_STREAM_CHUNK_SIZE):
                chunk = response[i:i + QWEN_STREAM_CHUNK_SIZE]
                if chunk:
                    yield create_chunk_response(chunk)
                    # Small delay to simulate real streaming
                    await asyncio.sleep(QWEN_STREAM_DELAY)
            
            yield create_done_response()
            
        except Exception as e:
            logger.error(f"Error in Qwen stream generation: {e}", exc_info=True)
            yield create_error_response(str(e))
    
    def is_available(self) -> bool:
        """
        Check if QwenVLM is available.
        This is a lightweight check - doesn't actually load the model.
        Model loading happens lazily on first use.
        """
        if not QWEN_AVAILABLE:
            return False
        
        # Just check if we can import and create the adapter
        # Don't actually load the model here (that's expensive)
        return True
    
    def is_model_loaded(self) -> bool:
        """Check if the model is actually loaded (for runtime checks)."""
        return self._initialized and self.qwen_vlm is not None and self.qwen_vlm.model is not None

