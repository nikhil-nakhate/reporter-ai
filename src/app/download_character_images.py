#!/usr/bin/env python
"""
Script to download character images and save them to character/img folders.
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.image_service import ImageService
from app.services.character_service import CharacterService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download images for all available characters."""
    image_service = ImageService()
    character_service = CharacterService()
    
    # Get all available personas
    personas = character_service.get_available_personas()
    
    logger.info(f"Found {len(personas)} characters to process")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    import time
    
    for persona in personas:
        character_id = persona["id"]
        character_name = persona["name"]
        
        logger.info(f"\nProcessing: {character_name} ({character_id})")
        
        # Check if image already exists
        existing_image = image_service.get_image_path(character_id)
        if existing_image:
            logger.info(f"  Image already exists: {existing_image}")
            skipped += 1
            continue
        
        # Download image
        try:
            image_path = image_service.download_character_image(character_id)
            if image_path:
                logger.info(f"  ✓ Successfully downloaded: {image_path}")
                downloaded += 1
            else:
                logger.warning(f"  ✗ Failed to download image (no URL available)")
                failed += 1
        except Exception as e:
            logger.error(f"  ✗ Error downloading image: {e}")
            failed += 1
        
        # Add delay to avoid rate limiting
        if downloaded > 0 or failed > 0:
            time.sleep(1)
    
    logger.info("\n" + "="*50)
    logger.info(f"Download Summary:")
    logger.info(f"  Downloaded: {downloaded}")
    logger.info(f"  Skipped (already exists): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info("="*50)


if __name__ == "__main__":
    main()

