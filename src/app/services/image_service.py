"""
Service for downloading and managing character images.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List
import requests
from PIL import Image
import io

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ImageService:
    """Service for downloading and managing character images."""
    
    def __init__(self):
        """Initialize the image service."""
        self.base_dir = Path(__file__).parent.parent.parent / "characters"
        self._character_image_urls = {
            "palki": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Palki_Sharma_Upadhyay.jpg/800px-Palki_Sharma_Upadhyay.jpg",
            "paul_graham": "https://live.staticflickr.com/65535/51241566043_0cdf9c5e4c_b.jpg",  # Alternative source
            "tony_stark": "https://upload.wikimedia.org/wikipedia/en/c/c6/Iron_Man_bleeding_edge.jpg",
            "yann": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Yann_LeCun.jpg/800px-Yann_LeCun.jpg",
            "donald_trump": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Donald_Trump_official_portrait.jpg/800px-Donald_Trump_official_portrait.jpg",
            "barack_obama": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/800px-President_Barack_Obama.jpg",
            "elon_musk": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/800px-Elon_Musk_Royal_Society_%28crop2%29.jpg",
            "joe_biden": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Joe_Biden_presidential_portrait.jpg/800px-Joe_Biden_presidential_portrait.jpg",
            "bill_gates": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Bill_Gates_2018.jpg/800px-Bill_Gates_2018.jpg",
            "oprah_winfrey": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Oprah_Winfrey_at_the_2011_TCA_Press_Tour.jpg/800px-Oprah_Winfrey_at_the_2011_TCA_Press_Tour.jpg",
            "mark_zuckerberg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg/800px-Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg",
        }
    
    def get_image_path(self, character_id: str) -> Optional[str]:
        """
        Get the image path for a character.
        
        Args:
            character_id: ID of the character
            
        Returns:
            Path to the character image or None if not found
        """
        image_dir = self.base_dir / character_id / "img"
        if not image_dir.exists():
            return None
        
        # Look for image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
        if image_files:
            return str(image_files[0])
        
        return None
    
    def download_character_image(self, character_id: str, image_url: Optional[str] = None) -> Optional[str]:
        """
        Download an image for a character from the internet.
        
        Args:
            character_id: ID of the character
            image_url: Optional URL to download from. If not provided, uses default URL.
            
        Returns:
            Path to the downloaded image or None if download failed
        """
        if image_url is None:
            image_url = self._character_image_urls.get(character_id)
        
        if not image_url:
            logger.warning(f"No image URL found for character: {character_id}")
            return None
        
        # Create img directory for character
        image_dir = self.base_dir / character_id / "img"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Download image
        try:
            logger.info(f"Downloading image for {character_id} from {image_url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(image_url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Verify it's an image
            image = Image.open(io.BytesIO(response.content))
            image = image.convert("RGB")
            
            # Save image
            image_path = image_dir / f"{character_id}_image.jpg"
            image.save(image_path, "JPEG", quality=95)
            logger.info(f"Successfully downloaded and saved image to {image_path}")
            
            return str(image_path)
        except Exception as e:
            logger.error(f"Error downloading image for {character_id}: {e}")
            return None
    
    def ensure_character_image(self, character_id: str) -> Optional[str]:
        """
        Ensure a character has an image, downloading if necessary.
        
        Args:
            character_id: ID of the character
            
        Returns:
            Path to the character image or None if not available
        """
        # Check if image already exists
        image_path = self.get_image_path(character_id)
        if image_path:
            logger.info(f"Image already exists for {character_id}: {image_path}")
            return image_path
        
        # Download if not exists
        logger.info(f"No image found for {character_id}, attempting to download...")
        return self.download_character_image(character_id)

