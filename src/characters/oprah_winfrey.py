from typing import List, Optional, Dict, Any
import yaml
import logging
from pathlib import Path
from .base import CharacterBase

logger = logging.getLogger(__name__)


class OprahWinfrey(CharacterBase):
    """
    Character class for Oprah Winfrey.
    
    This class extends CharacterBase with specific configuration for Oprah Winfrey.
    """
    
    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        voice_samples: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
        persona_llm_ckpt_path: Optional[str] = None,
        video_paths: Optional[List[str]] = None,
        yaml_path: Optional[str] = None
    ):
        """
        Initialize Oprah Winfrey character.
        
        Args:
            image_paths: List of paths to Oprah Winfrey images
            voice_samples: List of paths to Oprah Winfrey voice sample audio files
            dataset_path: Path to the Oprah Winfrey dataset folder
            persona_llm_ckpt_path: Path to the Oprah Winfrey persona LLM checkpoint
            video_paths: List of paths to Oprah Winfrey videos
            yaml_path: Path to the oprah_winfrey.yaml config file (defaults to characters/oprah_winfrey/oprah_winfrey.yaml)
        """
        super().__init__(
            image_paths=image_paths,
            voice_samples=voice_samples,
            dataset_path=dataset_path,
            persona_llm_ckpt_path=persona_llm_ckpt_path,
            video_paths=video_paths
        )
        self.name = "Oprah Winfrey"
        
        # Load YAML configuration
        if yaml_path is None:
            # Default to characters/oprah_winfrey/oprah_winfrey.yaml relative to this file
            current_dir = Path(__file__).parent
            yaml_path = current_dir / "oprah_winfrey" / "oprah_winfrey.yaml"
        
        self.yaml_path = yaml_path
        self.config: Dict[str, Any] = {}
        self._load_config()
        
        # Auto-load voice samples if not provided
        if not self.voice_samples:
            self._load_voice_samples()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.yaml_path or not Path(self.yaml_path).exists():
            return
        
        try:
            with open(self.yaml_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config from {self.yaml_path}: {e}")
            self.config = {}
    
    def get_system_prompt(self) -> str:
        """Get the system prompt from the YAML config."""
        return self.config.get("system_prompt", "")
    
    def get_persona_prompt(self) -> str:
        """Get the persona prompt from the YAML config."""
        return self.config.get("persona_prompt", "")
    
    def get_task_prompt(self) -> str:
        """Get the task prompt from the YAML config."""
        return self.config.get("task_prompt", "")
    
    def get_full_prompt(self) -> str:
        """Get the combined system and persona prompt."""
        system = self.get_system_prompt()
        persona = self.get_persona_prompt()
        
        if system and persona:
            return f"{system}\n\n{persona}"
        elif system:
            return system
        elif persona:
            return persona
        else:
            return ""
    
    def _load_voice_samples(self):
        """Auto-load voice samples from the character's voice directory."""
        current_dir = Path(__file__).parent
        voice_dir = current_dir / "oprah_winfrey" / "voice"
        
        if voice_dir.exists():
            # Look for voice sample files
            voice_files = list(voice_dir.glob("*.wav")) + list(voice_dir.glob("*.mp3"))
            if voice_files:
                self.voice_samples = [str(f) for f in voice_files]
                logger.info(f"Auto-loaded {len(self.voice_samples)} voice samples for Oprah Winfrey")
    
    def __repr__(self) -> str:
        """String representation of the Oprah Winfrey character."""
        return (
            f"OprahWinfrey("
            f"name={self.name}, "
            f"image_paths={len(self.image_paths)} images, "
            f"voice_samples={len(self.voice_samples)} samples, "
            f"dataset_path={self.dataset_path}, "
            f"persona_llm_ckpt_path={self.persona_llm_ckpt_path}, "
            f"video_paths={len(self.video_paths)} videos, "
            f"config_loaded={bool(self.config)}"
            f")"
        )
