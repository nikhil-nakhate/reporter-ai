"""
Script to explore the voices-celebrities dataset and find available celebrities.
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset

def explore_dataset():
    """Explore the dataset to find all available celebrities."""
    print("Loading dataset...")
    try:
        dataset = load_dataset("sdialog/voices-celebrities", split="train")
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        print("\n" + "="*60)
        print("Available Celebrities:")
        print("="*60)
        
        # Collect unique celebrity names
        celebrities = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            # Try different possible field names for the celebrity name
            name = None
            if 'name' in sample:
                name = sample['name']
            elif 'speaker' in sample:
                name = sample['speaker']
            elif 'celebrity' in sample:
                name = sample['celebrity']
            elif 'person' in sample:
                name = sample['person']
            
            # If no name field, try to get from path or other metadata
            if not name:
                # Check if there's a path that might contain the name
                if 'path' in sample:
                    path = sample['path']
                    # Extract name from path if possible
                    name = os.path.basename(str(path)).split('_')[0] if path else None
                elif 'file' in sample:
                    file = sample['file']
                    name = os.path.basename(str(file)).split('_')[0] if file else None
            
            # Normalize name
            if name:
                name = str(name).strip()
                if name and name not in celebrities:
                    celebrities[name] = i
                    print(f"  {len(celebrities)}. {name} (index: {i})")
        
        print(f"\nTotal unique celebrities found: {len(celebrities)}")
        print("\n" + "="*60)
        print("Celebrity to Index Mapping:")
        print("="*60)
        for name, idx in sorted(celebrities.items()):
            print(f"  '{name}': {idx}")
        
        # Also print first sample structure to understand the data
        if len(dataset) > 0:
            print("\n" + "="*60)
            print("Sample structure (first item):")
            print("="*60)
            first_sample = dataset[0]
            for key in first_sample.keys():
                value = first_sample[key]
                if isinstance(value, dict):
                    print(f"  {key}: dict with keys: {list(value.keys())}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} of length {len(value)}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        return celebrities
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    explore_dataset()

