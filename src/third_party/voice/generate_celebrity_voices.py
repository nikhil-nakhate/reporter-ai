"""
Script to copy voice samples for multiple celebrities from HuggingFace cache.
The voice samples are already generated and stored in the HuggingFace cache at:
~/.cache/huggingface/datasets/sdialog___voices-celebrities

This script copies them from the cache to organized folders for use with the TTS system.

IMPORTANT: This script should be run with the 'reporter' conda environment activated.
Run: conda activate reporter
Or: conda run -n reporter python generate_celebrity_voices.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from datasets import load_dataset

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# List of celebrities to generate voice samples for
# Celebrities available in the sdialog/voices-celebrities dataset
CELEBRITIES = [
    # From the dataset (indices discovered via explore_celebrities.py)
    "andrew_tate",      # index: 0
    "barack_obama",     # index: 1
    "bill_gates",       # index: 2
    "donald_trump",     # index: 3
    "elon_musk",        # index: 4
    "greta_thunberg",   # index: 5
    "hillary_clinton",  # index: 6
    "jk_rowling",       # index: 7
    "jensen_huang",     # index: 8
    "joe_biden",        # index: 9
    "kamala_harris",    # index: 10
    "mark_zuckerberg",  # index: 11
    "oprah_winfrey",    # index: 12
    # Custom characters (not in dataset, need manual voice samples)
    "palki_sharma",
    "paul_graham", 
    "tony_stark",
    "yann_lecun",
]

# Map celebrity names to dataset indices
# Based on exploration of sdialog/voices-celebrities dataset
CELEBRITY_DATASET_MAP = {
    "andrew_tate": 0,
    "barack_obama": 1,
    "bill_gates": 2,
    "donald_trump": 3,
    "elon_musk": 4,
    "greta_thunberg": 5,
    "hillary_clinton": 6,
    "jk_rowling": 7,
    "jensen_huang": 8,
    "joe_biden": 9,
    "kamala_harris": 10,
    "mark_zuckerberg": 11,
    "oprah_winfrey": 12,
    # Custom characters - will need manual voice samples or different dataset
    # "palki_sharma": None,
    # "paul_graham": None,
    # "tony_stark": None,
    # "yann_lecun": None,
}


def find_celebrity_in_dataset(dataset, celebrity_name: str):
    """
    Find a celebrity in the dataset by searching through metadata.
    
    Args:
        dataset: The loaded dataset
        celebrity_name: Name of the celebrity to find
        
    Returns:
        Tuple of (index, audio_data) or (None, None) if not found
    """
    # Normalize celebrity name for searching
    search_terms = [
        celebrity_name.lower().replace('_', ' '),
        celebrity_name.lower().replace('_', ''),
        celebrity_name.lower(),
    ]
    
    # Also try first and last name separately
    parts = celebrity_name.lower().replace('_', ' ').split()
    if len(parts) > 1:
        search_terms.extend(parts)
    
    print(f"Searching for '{celebrity_name}' using terms: {search_terms}")
    
    for i, sample in enumerate(dataset):
        # Check various possible fields for the name
        sample_text = ""
        for field in ['name', 'speaker', 'celebrity', 'person', 'label', 'id']:
            if field in sample:
                value = str(sample[field]).lower()
                sample_text += f" {value}"
        
        # Check if any search term matches
        for term in search_terms:
            if term in sample_text:
                print(f"  ✓ Found at index {i}: {sample_text.strip()}")
                return i, sample.get('audio')
    
    return None, None


def get_celebrity_from_dataset(dataset, celebrity_name: str, index: int = None):
    """
    Get a celebrity sample from the dataset.
    
    Args:
        dataset: The loaded dataset
        celebrity_name: Name of the celebrity
        index: Optional specific index to use
        
    Returns:
        Audio data dictionary or None
    """
    if index is not None:
        try:
            sample = dataset[index]
            return sample.get('audio')
        except Exception as e:
            print(f"Error accessing index {index}: {e}")
            # Fall through to search
    
    # Try to find by name in metadata
    found_index, audio_data = find_celebrity_in_dataset(dataset, celebrity_name)
    if audio_data:
        return audio_data
    
    # Fallback: use first available sample
    print(f"Warning: Could not find specific sample for {celebrity_name}, using first available")
    try:
        return dataset[0].get('audio')
    except:
        return None


def generate_voice_sample(celebrity_name: str, output_dir: str, dataset_index: int = None):
    """
    Copy voice sample for a celebrity from HuggingFace cache to character directory.
    The voice samples are already generated and stored in the HuggingFace cache.
    
    Args:
        celebrity_name: Name of the celebrity
        output_dir: Directory to save the voice sample
        dataset_index: Optional specific dataset index to use
    """
    print(f"\n{'='*60}")
    print(f"Copying voice sample for: {celebrity_name}")
    print(f"{'='*60}")
    
    try:
        # Load dataset (this accesses the cached files)
        print("Loading dataset from cache...")
        dataset = load_dataset("sdialog/voices-celebrities", split="train")
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        
        # Get the index to use - prefer provided index, then map, then search
        index = dataset_index or CELEBRITY_DATASET_MAP.get(celebrity_name)
        
        # Get audio sample (will search if index not found)
        print(f"Extracting audio sample from cache...")
        audio_data = get_celebrity_from_dataset(dataset, celebrity_name, index)
        
        if not audio_data:
            print(f"❌ Error: Could not extract audio for {celebrity_name}")
            return False
        
        # Get the cached audio file path
        cached_audio_path = audio_data.get('path')
        audio_array = audio_data.get('array')
        sampling_rate = audio_data.get('sampling_rate')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{celebrity_name}_voice_sample.wav")
        
        # Try to copy directly from cache if path exists and is a file
        if cached_audio_path and os.path.exists(cached_audio_path) and os.path.isfile(cached_audio_path):
            print(f"  📁 Source (cache): {cached_audio_path}")
            print(f"  📁 Destination: {output_path}")
            import shutil
            shutil.copy2(cached_audio_path, output_path)
            
            # Verify the copy was successful
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"✅ Voice sample copied successfully!")
                print(f"   File size: {file_size:.2f} MB")
                print(f"   Saved to: {output_path}")
            else:
                print(f"❌ Error: Copy failed - destination file not found")
                return False
        else:
            # Fallback: extract from array and save
            print(f"  Extracting from audio array...")
            if audio_array is None or sampling_rate is None:
                print(f"❌ Error: Audio data incomplete for {celebrity_name}")
                return False
            
            # Ensure the array is in the correct format
            if audio_array.dtype != np.float32 and audio_array.dtype != np.float64:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize audio to prevent clipping
            max_val = np.abs(audio_array).max()
            if max_val > 1.0:
                audio_array = audio_array / max_val
            
            # Save the audio file
            sf.write(output_path, audio_array, sampling_rate)
            print(f"✅ Voice sample saved to: {output_path}")
        
        # Get file info
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            if audio_array is not None and sampling_rate is not None:
                duration = len(audio_array) / sampling_rate
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Sampling rate: {sampling_rate} Hz")
            print(f"   File size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying voice sample for {celebrity_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Generate voice samples for all celebrities."""
    # Base directory for voice samples (relative to characters folder)
    base_output_dir = Path(__file__).parent.parent.parent / "characters"
    
    print("\n" + "="*60)
    print("Celebrity Voice Sample Copier")
    print("Copying from HuggingFace cache to character directories")
    print("="*60)
    
    success_count = 0
    failed_celebrities = []
    
    for celebrity in CELEBRITIES:
        # Create celebrity-specific directory
        celebrity_dir = base_output_dir / celebrity / "voice"
        celebrity_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate voice sample
        if generate_voice_sample(celebrity, str(celebrity_dir)):
            success_count += 1
        else:
            failed_celebrities.append(celebrity)
    
    # Summary
    print("\n" + "="*60)
    print("Generation Summary")
    print("="*60)
    print(f"✅ Successfully generated: {success_count}/{len(CELEBRITIES)}")
    
    if failed_celebrities:
        print(f"❌ Failed celebrities: {', '.join(failed_celebrities)}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

