from typing import List, Optional
from .base import CharacterBase


class PaulGraham(CharacterBase):
    """
    Character class for Paul Graham.
    
    This class extends CharacterBase with specific configuration for Paul Graham.
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
        Initialize Paul Graham character.
        
        Args:
            image_paths: List of paths to Paul Graham images
            voice_samples: List of paths to Paul Graham voice sample audio files
            dataset_path: Path to the Paul Graham dataset folder
            persona_llm_ckpt_path: Path to the Paul Graham persona LLM checkpoint
            video_paths: List of paths to Paul Graham videos
        """
        super().__init__(
            image_paths=image_paths,
            voice_samples=voice_samples,
            dataset_path=dataset_path,
            persona_llm_ckpt_path=persona_llm_ckpt_path,
            video_paths=video_paths
        )
        self.name = "Paul Graham"
    
    def __repr__(self) -> str:
        """String representation of the Paul Graham character."""
        return (
            f"PaulGraham("
            f"name={self.name}, "
            f"image_paths={len(self.image_paths)} images, "
            f"voice_samples={len(self.voice_samples)} samples, "
            f"dataset_path={self.dataset_path}, "
            f"persona_llm_ckpt_path={self.persona_llm_ckpt_path}, "
            f"video_paths={len(self.video_paths)} videos"
            f")"
        )

