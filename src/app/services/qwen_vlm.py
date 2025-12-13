"""
Qwen VLM implementation for text generation.
Based on the robobrains implementation.
"""
import io
import os
import shutil
import base64
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class QwenVLM:
    """Qwen Vision-Language Model for text generation."""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", models_dir: str = None):
        """
        Initialize Qwen VLM.
        
        Args:
            model_id: HuggingFace model identifier
            models_dir: Directory to store models (defaults to ~/Models)
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set up models directory
        if models_dir is None:
            models_dir = os.path.expanduser("~/Models")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific directory
        model_name = model_id.replace("/", "--")
        self.model_dir = self.models_dir / model_name
        
        logger.info(f"Loading Qwen VLM: {model_id} on {self.device}")
        logger.info(f"Models directory: {self.model_dir}")
        
        try:
            # Check if model exists in ~/Models, if not download and move it
            self._ensure_model_in_models_dir()
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load from local directory if files exist, otherwise load from HuggingFace
            # (which will download, and we'll move files after)
            model_path = str(self.model_dir) if (self.model_dir.exists() and any(self.model_dir.iterdir())) else self.model_id
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                torch_dtype="auto", 
                device_map="auto",
                local_files_only=False  # Allow downloading missing files
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=False
            )
            logger.info("Qwen VLM loaded successfully")
            
            # If we loaded from HuggingFace (not local), move files to ~/Models after download
            if model_path == self.model_id:
                logger.info("Model was downloaded from HuggingFace, moving files to ~/Models...")
                # The files are now in cache, we'll move them in the background
                # For now, just log - the _ensure_model_in_models_dir will handle it on next run
        except Exception as e:
            logger.error(f"Failed to load Qwen VLM: {e}")
            self.model = None
            self.processor = None
    
    def _ensure_model_in_models_dir(self):
        """
        Ensure model files are in ~/Models directory.
        Downloads if not present, or moves from cache if already downloaded.
        """
        # Check if model directory already exists and has files
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            # Check if it has the essential files (safetensors or pytorch_model files)
            has_model_files = any(
                f.suffix in ['.safetensors', '.bin'] or f.name.startswith('model')
                for f in self.model_dir.rglob('*')
            )
            if has_model_files:
                logger.info(f"Model found in {self.model_dir}, using existing files")
                return
        
        logger.info(f"Model not found in {self.model_dir}, downloading...")
        
        # Download model to cache first, then move to ~/Models
        try:
            # Use snapshot_download to get all files (downloads to HuggingFace cache)
            cache_dir = snapshot_download(
                repo_id=self.model_id,
                cache_dir=None,  # Use default HuggingFace cache
                local_files_only=False
            )
            
            logger.info(f"Model downloaded to cache at {cache_dir}, moving to {self.model_dir}...")
            
            # Create destination directory
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from cache to ~/Models
            if os.path.exists(cache_dir):
                for item in os.listdir(cache_dir):
                    src = os.path.join(cache_dir, item)
                    dst = os.path.join(self.model_dir, item)
                    
                    if os.path.isdir(src):
                        # Copy directory
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        logger.debug(f"Copied directory: {item}")
                    else:
                        # Copy file
                        shutil.copy2(src, dst)
                        logger.debug(f"Copied file: {item}")
                
                logger.info(f"✅ Model files successfully moved to {self.model_dir}")
                logger.info(f"   Model files are now in: {self.model_dir}")
            else:
                logger.warning(f"Cache directory {cache_dir} does not exist")
                
        except Exception as e:
            logger.warning(f"Failed to download/move model: {e}")
            logger.info("Will try to load from HuggingFace cache or download on-the-fly")
            logger.debug(f"Error details: {e}", exc_info=True)
            # Create directory anyway for future use
            self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _move_downloaded_model_to_models_dir(self):
        """
        Move model files from HuggingFace cache to ~/Models after download.
        This is called after the model has been loaded to ensure files are moved.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # Get the cache location where the model was downloaded
            from transformers.utils import TRANSFORMERS_CACHE
            cache_base = os.path.expanduser(TRANSFORMERS_CACHE) if TRANSFORMERS_CACHE else os.path.expanduser("~/.cache/huggingface")
            
            # Find the model in cache
            model_cache_path = None
            for root, dirs, files in os.walk(cache_base):
                if self.model_id.replace("/", "--") in root or any(".safetensors" in f for f in files):
                    model_cache_path = root
                    break
            
            if model_cache_path and os.path.exists(model_cache_path):
                logger.info(f"Found model in cache at {model_cache_path}, moving to {self.model_dir}...")
                self.model_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files
                for item in os.listdir(model_cache_path):
                    src = os.path.join(model_cache_path, item)
                    dst = os.path.join(self.model_dir, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                
                logger.info(f"✅ Model files moved to {self.model_dir}")
        except Exception as e:
            logger.debug(f"Could not move model files (this is okay): {e}")
    
    def generate_response(self, image_base64: str, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response with image input.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if not self.model or not self.processor:
            return "Error: VLM model not loaded."

        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            )

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)

            # Inference
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]

        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            return f"Error generating response: {e}"
    
    def generate_text_response(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 2048) -> str:
        """
        Generate a text-only response (no image input).
        Uses a minimal blank image as placeholder since the model expects image input.
        
        Args:
            prompt: Text prompt
            system_prompt: Optional system prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        if not self.model or not self.processor:
            return "Error: VLM model not loaded."
        
        try:
            # Create a minimal blank image (1x1 pixel) as placeholder
            blank_image = Image.new('RGB', (1, 1), color='white')
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": blank_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Inference with configurable token limit
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
        
        except Exception as e:
            logger.error(f"VLM text generation failed: {e}")
            return f"Error generating text response: {e}"

