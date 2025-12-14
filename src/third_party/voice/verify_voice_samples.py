"""
Script to verify that voice samples are correctly copied and mapped to celebrities.
"""
import os
from pathlib import Path
from datasets import load_dataset

# Map of character IDs to dataset indices and expected names
CELEBRITY_VERIFICATION = {
    "andrew_tate": {"index": 0, "expected_name": "Andrew Tate"},
    "barack_obama": {"index": 1, "expected_name": "Barack Obama"},
    "bill_gates": {"index": 2, "expected_name": "Bill Gates"},
    "donald_trump": {"index": 3, "expected_name": "Donald Trump"},
    "elon_musk": {"index": 4, "expected_name": "Elon Musk"},
    "greta_thunberg": {"index": 5, "expected_name": "Greta Thunberg"},
    "hillary_clinton": {"index": 6, "expected_name": "Hillary Clinton"},
    "jk_rowling": {"index": 7, "expected_name": "J.K. Rowling"},
    "jensen_huang": {"index": 8, "expected_name": "Jensen Huang"},
    "joe_biden": {"index": 9, "expected_name": "Joe Biden"},
    "kamala_harris": {"index": 10, "expected_name": "Kamala Harris"},
    "mark_zuckerberg": {"index": 11, "expected_name": "Mark Zuckerberg"},
    "oprah_winfrey": {"index": 12, "expected_name": "Oprah Winfrey"},
}

def verify_voice_samples():
    """Verify that voice samples are correctly copied and mapped."""
    print("\n" + "="*60)
    print("Voice Sample Verification")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent.parent / "characters"
    dataset = load_dataset("sdialog/voices-celebrities", split="train")
    
    verified = []
    missing = []
    mismatched = []
    
    for character_id, info in CELEBRITY_VERIFICATION.items():
        # Check if file exists
        voice_file = base_dir / character_id / "voice" / f"{character_id}_voice_sample.wav"
        
        # Get expected data from dataset
        sample = dataset[info["index"]]
        expected_name = sample.get("name", "Unknown")
        
        if voice_file.exists():
            file_size = voice_file.stat().st_size / (1024 * 1024)  # MB
            # Verify the name matches
            if expected_name == info["expected_name"]:
                verified.append({
                    "character": character_id,
                    "file": str(voice_file),
                    "size_mb": file_size,
                    "dataset_name": expected_name,
                    "index": info["index"]
                })
                print(f"✅ {character_id:20s} -> {expected_name:20s} ({file_size:.2f} MB)")
            else:
                mismatched.append({
                    "character": character_id,
                    "expected": info["expected_name"],
                    "found": expected_name
                })
                print(f"⚠️  {character_id:20s} -> Name mismatch: expected {info['expected_name']}, found {expected_name}")
        else:
            missing.append(character_id)
            print(f"❌ {character_id:20s} -> File not found: {voice_file}")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✅ Verified: {len(verified)}/{len(CELEBRITY_VERIFICATION)}")
    print(f"❌ Missing: {len(missing)}")
    print(f"⚠️  Mismatched: {len(mismatched)}")
    
    if verified:
        print("\nVerified voice samples:")
        for v in verified:
            print(f"  - {v['character']}: {v['dataset_name']} (index {v['index']}, {v['size_mb']:.2f} MB)")
    
    if missing:
        print(f"\nMissing voice samples: {', '.join(missing)}")
        print("Run: python src/third_party/voice/generate_celebrity_voices.py")
    
    return len(verified) == len(CELEBRITY_VERIFICATION)

if __name__ == "__main__":
    verify_voice_samples()

