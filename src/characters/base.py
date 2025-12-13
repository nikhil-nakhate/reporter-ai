from abc import ABC
from typing import List, Optional
from pathlib import Path


class CharacterBase(ABC):
    """
    Base class for character definitions.
    
    Attributes:
        image_paths: List of paths to character images
        voice_samples: List of paths to voice sample audio files
        dataset_path: Path to the dataset folder
        persona_llm_ckpt_path: Path to the persona LLM checkpoint
        video_paths: List of paths to character videos
    """
    
    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        voice_samples: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
        persona_llm_ckpt_path: Optional[str] = None,
        video_paths: Optional[List[str]] = None
    ):
        """
        Initialize the character base class.
        
        Args:
            image_paths: List of paths to character images
            voice_samples: List of paths to voice sample audio files
            dataset_path: Path to the dataset folder
            persona_llm_ckpt_path: Path to the persona LLM checkpoint
            video_paths: List of paths to character videos
        """
        self.image_paths = image_paths or []
        self.voice_samples = voice_samples or []
        self.dataset_path = dataset_path
        self.persona_llm_ckpt_path = persona_llm_ckpt_path
        self.video_paths = video_paths or []
    
    def validate_paths(self) -> bool:
        """
        Validate that all specified paths exist.
        
        Returns:
            True if all paths are valid, False otherwise
        """
        all_valid = True
        
        # Validate image paths
        for img_path in self.image_paths:
            if not Path(img_path).exists():
                print(f"Warning: Image path does not exist: {img_path}")
                all_valid = False
        
        # Validate voice sample paths
        for voice_path in self.voice_samples:
            if not Path(voice_path).exists():
                print(f"Warning: Voice sample path does not exist: {voice_path}")
                all_valid = False
        
        # Validate dataset path
        if self.dataset_path and not Path(self.dataset_path).exists():
            print(f"Warning: Dataset path does not exist: {self.dataset_path}")
            all_valid = False
        
        # Validate persona LLM checkpoint path
        if self.persona_llm_ckpt_path and not Path(self.persona_llm_ckpt_path).exists():
            print(f"Warning: Persona LLM checkpoint path does not exist: {self.persona_llm_ckpt_path}")
            all_valid = False
        
        # Validate video paths
        for video_path in self.video_paths:
            if not Path(video_path).exists():
                print(f"Warning: Video path does not exist: {video_path}")
                all_valid = False
        
        return all_valid

