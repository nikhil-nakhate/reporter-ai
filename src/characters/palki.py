from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path
from .base import CharacterBase


class Palki(CharacterBase):
    """
    Character class for Palki Sharma.
    
    This class extends CharacterBase with specific configuration for Palki Sharma,
    a renowned news anchor and managing editor known for her analytical approach
    to international affairs.
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
        Initialize Palki Sharma character.
        
        Args:
            image_paths: List of paths to Palki Sharma images
            voice_samples: List of paths to Palki Sharma voice sample audio files
            dataset_path: Path to the Palki Sharma dataset folder
            persona_llm_ckpt_path: Path to the Palki Sharma persona LLM checkpoint
            video_paths: List of paths to Palki Sharma videos
            yaml_path: Path to the palki.yaml config file (defaults to characters/palki/palki.yaml)
        """
        super().__init__(
            image_paths=image_paths,
            voice_samples=voice_samples,
            dataset_path=dataset_path,
            persona_llm_ckpt_path=persona_llm_ckpt_path,
            video_paths=video_paths
        )
        self.name = "Palki Sharma"
        
        # Load YAML configuration
        if yaml_path is None:
            # Default to characters/palki/palki.yaml relative to this file
            current_dir = Path(__file__).parent
            yaml_path = current_dir / "palki" / "palki.yaml"
        
        self.yaml_path = yaml_path
        self.config: Dict[str, Any] = {}
        self._load_config()
    
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
    
    def __repr__(self) -> str:
        """String representation of the Palki Sharma character."""
        return (
            f"Palki("
            f"name={self.name}, "
            f"image_paths={len(self.image_paths)} images, "
            f"voice_samples={len(self.voice_samples)} samples, "
            f"dataset_path={self.dataset_path}, "
            f"persona_llm_ckpt_path={self.persona_llm_ckpt_path}, "
            f"video_paths={len(self.video_paths)} videos, "
            f"config_loaded={bool(self.config)}"
            f")"
        )
