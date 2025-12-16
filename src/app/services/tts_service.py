"""
Service for Text-to-Speech generation using celebrity voice samples.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Apply compatibility patch for newer transformers versions
# This must be imported BEFORE TTS to fix BeamSearchScorer import issue
try:
    from app.services import tts_compat
except ImportError:
    # If running as standalone, try relative import
    try:
        from . import tts_compat
    except ImportError:
        pass  # Compatibility patch not critical if import fails

import torch
from TTS.api import TTS
from pydub import AudioSegment

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import DEFAULT_TTS_MODEL

logger = logging.getLogger(__name__)


class TTSService:
    """Service for generating audio from text using TTS with celebrity voice samples."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the TTS service.
        
        Args:
            model_path: Path to the TTS model (defaults to DEFAULT_TTS_MODEL from config)
                       Options:
                       - "tts_models/multilingual/multi-dataset/xtts_v2" (recommended)
                       - "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"
        """
        self.model_path = model_path or DEFAULT_TTS_MODEL
        self.model: Optional[TTS] = None
        # Force CPU-only to save GPU memory for video service
        self.device = "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TTS model."""
        try:
            logger.info(f"Initializing TTS model: {self.model_path} on device: {self.device}")
            self.model = TTS(self.model_path, progress_bar=False).to(self.device)
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            logger.warning("TTS service will not be available. Audio generation will be disabled.")
            
            # Check for specific error types and provide targeted fixes
            error_str = str(e).lower()
            if "beamsearchscorer" in error_str or "cannot import name" in error_str:
                logger.warning("Transformers version compatibility issue detected.")
                logger.warning("TTS 0.22.0 requires transformers <4.40.0, but echomimic_v3 requires >=4.46.2")
                logger.warning("Solutions:")
                logger.warning("  1. Use Python 3.11+ where TTS may work with newer transformers")
                logger.warning("  2. Install transformers 4.35.0 in a separate environment for TTS only")
                logger.warning("  3. Wait for TTS update that supports newer transformers")
            else:
                logger.warning("Common fixes:")
                logger.warning("  - PyTorch version: pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1")
                logger.warning("  - Transformers: pip install transformers==4.35.0")
                logger.warning("  - HuggingFace Hub: pip install 'huggingface_hub>=0.20.0,<1.0'")
            
            self.model = None
            # Don't raise - allow app to continue without TTS
    
    def _chunk_text(self, text: str, max_chars: int = 500) -> List[str]:
        """
        Split text into chunks for TTS processing.
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds max length, save current chunk
            if len(current_chunk) + len(paragraph) + 2 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
            else:
                current_chunk += paragraph + "\n\n"
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_audio(
        self,
        text: str,
        voice_sample_path: str,
        output_path: Optional[str] = None,
        language: str = "en",
        save_chunks: bool = False
    ) -> str:
        """
        Generate audio from text using a voice sample.
        
        Args:
            text: Text to convert to speech
            voice_sample_path: Path to the reference voice sample
            output_path: Optional output path for the audio file
            language: Language code (default: "en")
            save_chunks: If True, keep chunk files instead of deleting them
            
        Returns:
            Path to the generated audio file
        """
        if not self.model:
            error_msg = (
                "TTS model not initialized. Check logs for initialization errors. "
                "This may be due to a transformers library compatibility issue. "
                "Try: pip install transformers==4.35.0"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if not os.path.exists(voice_sample_path):
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")
        
        # Verify and log voice sample details
        file_size = os.path.getsize(voice_sample_path) / (1024 * 1024)  # MB
        logger.info(f"Using voice sample: {voice_sample_path}")
        logger.info(f"Voice sample size: {file_size:.2f} MB")
        
        # Extract celebrity name from path for verification
        path_parts = Path(voice_sample_path).parts
        if "characters" in path_parts:
            char_idx = path_parts.index("characters")
            if char_idx + 1 < len(path_parts):
                celebrity_name = path_parts[char_idx + 1]
                logger.info(f"Voice sample for celebrity: {celebrity_name}")
        
        # Create temporary directory for chunks if output_path not specified
        if output_path is None:
            output_path = "/tmp/bulletin_audio.mp3"
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Chunk the text
        text_chunks = self._chunk_text(text)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        # Generate audio for each chunk
        chunk_files = []
        temp_dir = "/tmp/tts_chunks"
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, chunk in enumerate(text_chunks):
            chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
            try:
                logger.info(f"Generating audio for chunk {i+1}/{len(text_chunks)}")
                logger.info(f"Using voice sample: {voice_sample_path}")
                # XTTS v2 uses speaker_wav and file_path parameters
                self.model.tts_to_file(
                    text=chunk,
                    file_path=chunk_file,
                    speaker_wav=voice_sample_path,
                    language=language
                )
                chunk_files.append(chunk_file)
            except Exception as e:
                logger.error(f"Error generating audio for chunk {i}: {e}")
                # Continue with other chunks
                continue
        
        if not chunk_files:
            raise RuntimeError("Failed to generate any audio chunks")
        
        # Combine all chunks into one audio file
        logger.info(f"Combining {len(chunk_files)} audio chunks")
        final_audio = AudioSegment.empty()
        
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                segment = AudioSegment.from_file(chunk_file)
                final_audio += segment
        
        # Export final audio
        final_audio.export(output_path, format="mp3")
        logger.info(f"Audio generated successfully: {output_path}")
        
        # Clean up chunk files unless save_chunks is True
        if not save_chunks:
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass
        
        return output_path
    
    def generate_audio_chunks(
        self,
        text: str,
        voice_sample_path: str,
        output_dir: Optional[str] = None,
        language: str = "en"
    ) -> List[str]:
        """
        Generate audio chunks from text and return paths to individual chunk files.
        
        Args:
            text: Text to convert to speech
            voice_sample_path: Path to the reference voice sample
            output_dir: Optional output directory for chunk files
            language: Language code (default: "en")
            
        Returns:
            List of paths to generated audio chunk files
        """
        if not self.model:
            error_msg = (
                "TTS model not initialized. Check logs for initialization errors. "
                "This may be due to a transformers library compatibility issue. "
                "Try: pip install transformers==4.35.0"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if not os.path.exists(voice_sample_path):
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")
        
        if output_dir is None:
            output_dir = "/tmp/tts_chunks"
        os.makedirs(output_dir, exist_ok=True)
        
        # Chunk the text
        text_chunks = self._chunk_text(text)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        # Generate audio for each chunk
        chunk_files = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_file = os.path.join(output_dir, f"chunk_{i:03d}.mp3")
            try:
                logger.info(f"Generating audio for chunk {i+1}/{len(text_chunks)}")
                self.model.tts_to_file(
                    text=chunk,
                    file_path=chunk_file,
                    speaker_wav=voice_sample_path,
                    language=language
                )
                chunk_files.append(chunk_file)
            except Exception as e:
                logger.error(f"Error generating audio for chunk {i}: {e}")
                continue
        
        if not chunk_files:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"Generated {len(chunk_files)} audio chunks")
        return chunk_files
    
    def get_voice_sample_path(self, character_id: str) -> Optional[str]:
        """
        Get the voice sample path for a character.
        
        Args:
            character_id: ID of the character
            
        Returns:
            Path to the voice sample file or None if not found
        """
        # Map character IDs to their voice sample paths
        base_dir = Path(__file__).parent.parent.parent / "characters"
        
        # Try different possible file names (in order of preference)
        possible_names = [
            f"{character_id}_voice_sample.wav",  # Preferred: character-specific name
            "voice_sample.wav",
            "celebrity_voice_sample.wav",
        ]
        
        # Check character-specific voice directory
        voice_dir = base_dir / character_id / "voice"
        if voice_dir.exists():
            for name in possible_names:
                voice_path = voice_dir / name
                if voice_path.exists():
                    logger.info(f"Found voice sample for {character_id}: {voice_path}")
                    return str(voice_path)
            logger.warning(f"Voice directory exists for {character_id} but no voice sample file found in {voice_dir}")
        else:
            logger.warning(f"Voice directory does not exist for {character_id}: {voice_dir}")
        
        # Fallback: check if there's a voice sample in the third_party directory
        fallback_path = Path(__file__).parent.parent.parent / "third_party" / "voice" / "celebrity_voice_sample.wav"
        if fallback_path.exists():
            logger.info(f"Using fallback voice sample: {fallback_path}")
            return str(fallback_path)
        
        logger.error(f"No voice sample found for character: {character_id}")
        return None

