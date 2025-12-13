"""
Service for managing character personas.
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from characters import CharacterBase, PaulGraham, Palki

logger = logging.getLogger(__name__)


class CharacterService:
    """Service for managing and accessing character personas."""
    
    def __init__(self):
        """Initialize the character service."""
        self.characters: Dict[str, CharacterBase] = {}
        self._load_characters()
    
    def _load_characters(self):
        """Load all available characters."""
        # Register available characters
        characters_to_register = [
            ("palki", Palki),
            ("paul_graham", PaulGraham),
        ]
        
        for name, character_class in characters_to_register:
            try:
                character = character_class()
                self.characters[name] = character
                logger.info(f"Loaded character: {name}")
            except Exception as e:
                logger.warning(f"Failed to load character {name}: {e}")
    
    def get_available_personas(self) -> List[Dict[str, str]]:
        """Get list of available personas with their display names."""
        personas = []
        for key, character in self.characters.items():
            personas.append({
                "id": key,
                "name": getattr(character, "name", key.replace("_", " ").title())
            })
        return personas
    
    def persona_exists(self, persona_id: str) -> bool:
        """Check if a persona exists."""
        return persona_id in self.characters
    
    def get_character(self, persona_id: str) -> CharacterBase:
        """Get a character instance by ID."""
        if persona_id not in self.characters:
            raise ValueError(f"Persona '{persona_id}' not found")
        return self.characters[persona_id]

