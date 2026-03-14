#!/usr/bin/env python3
"""Download public domain speech samples for language detection validation.

This script downloads short speech samples in English, German, and Spanish
from LibriVox public domain audiobooks for use in validating and tuning
the language detection system.

Usage:
    python scripts/download_validation_audio.py

Audio samples are saved to validation_audio/ directory (gitignored).
The script is idempotent - it skips files that already exist.

Audio sources:
    - English: LibriVox public domain audiobook excerpt
    - German: LibriVox public domain audiobook excerpt
    - Spanish: LibriVox public domain audiobook excerpt

All audio is in the public domain (LibriVox only publishes public domain works).
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path


# Audio samples directory
AUDIO_DIR = Path(__file__).parent.parent / "validation_audio"

# Audio samples configuration
# NOTE: These URLs are placeholders. To use this script, replace with actual
# LibriVox or other public domain audio URLs.
SAMPLES = {
    "eng": {
        "url": "https://www.archive.org/download/PLACEHOLDER_ENGLISH_AUDIO",
        "filename": "eng_sample.mp3",
        "output": "eng_sample.wav",
        "description": "English speech sample (public domain LibriVox)",
        "source": "LibriVox - Example Book Title",
    },
    "deu": {
        "url": "https://www.archive.org/download/PLACEHOLDER_GERMAN_AUDIO",
        "filename": "deu_sample.mp3",
        "output": "deu_sample.wav",
        "description": "German speech sample (public domain LibriVox)",
        "source": "LibriVox - Example German Book Title",
    },
    "spa": {
        "url": "https://www.archive.org/download/PLACEHOLDER_SPANISH_AUDIO",
        "filename": "spa_sample.mp3",
        "output": "spa_sample.wav",
        "description": "Spanish speech sample (public domain LibriVox)",
        "source": "LibriVox - Example Spanish Book Title",
    },
    "fra": {
        "url": "https://www.archive.org/download/PLACEHOLDER_FRENCH_AUDIO",
        "filename": "fra_sample.mp3",
        "output": "fra_sample.wav",
        "description": "French speech sample (public domain LibriVox)",
        "source": "LibriVox - Example French Book Title",
    },
    "ita": {
        "url": "https://www.archive.org/download/PLACEHOLDER_ITALIAN_AUDIO",
        "filename": "ita_sample.mp3",
        "output": "ita_sample.wav",
        "description": "Italian speech sample (public domain LibriVox)",
        "source": "LibriVox - Example Italian Book Title",
    },
}


def download_sample(lang_code: str, info: dict) -> bool:
    """Download and convert a single audio sample.

    Args:
        lang_code: Language code (eng, deu, spa)
        info: Sample information dict

    Returns:
        True if successful or already exists, False on error
    """
    output_path = AUDIO_DIR / info["output"]

    # Skip if already exists
    if output_path.exists():
        print(f"✓ {lang_code.upper()}: {info['output']} already exists, skipping")
        return True

    print(f"Downloading {lang_code.upper()}: {info['description']}")
    print(f"  Source: {info['source']}")

    # Check if URL is placeholder
    if "PLACEHOLDER" in info["url"]:
        print(f"  WARNING: Placeholder URL detected. Please update with real audio URL.")
        print(f"  Suggested sources:")
        print(f"    - LibriVox (librivox.org) - public domain audiobooks")
        print(f"    - Mozilla Common Voice (commonvoice.mozilla.org) - CC-0 licensed")
        print(f"    - Internet Archive (archive.org) - public domain audio")
        return False

    try:
        # Download MP3
        mp3_path = AUDIO_DIR / info["filename"]
        print(f"  Downloading from: {info['url']}")
        urllib.request.urlretrieve(info["url"], mp3_path)
        print(f"  ✓ Downloaded to {mp3_path}")

        # Convert to WAV using ffmpeg
        print(f"  Converting to WAV...")
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(mp3_path),
                "-ar", "16000",  # 16kHz sample rate (standard for speech)
                "-ac", "1",       # mono
                "-y",             # overwrite
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR: ffmpeg conversion failed")
            print(f"  {result.stderr}")
            return False

        print(f"  ✓ Converted to {output_path}")

        # Remove temporary MP3
        mp3_path.unlink()
        print(f"  ✓ Cleaned up {mp3_path}")

        return True

    except urllib.error.URLError as e:
        print(f"  ERROR: Download failed: {e}")
        return False
    except FileNotFoundError:
        print(f"  ERROR: ffmpeg not found. Please install ffmpeg:")
        print(f"    Arch Linux: sudo pacman -S ffmpeg")
        print(f"    Ubuntu/Debian: sudo apt install ffmpeg")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Download all validation audio samples."""
    print("=" * 70)
    print("Validation Audio Download Script")
    print("=" * 70)
    print()
    print("This script downloads public domain speech samples for validating")
    print("language detection. All audio is from LibriVox public domain audiobooks.")
    print()

    # Create audio directory
    AUDIO_DIR.mkdir(exist_ok=True)
    print(f"Audio directory: {AUDIO_DIR}")
    print()

    # Download each sample
    success_count = 0
    for lang_code, info in SAMPLES.items():
        if download_sample(lang_code, info):
            success_count += 1
        print()

    # Summary
    print("=" * 70)
    print(f"Download complete: {success_count}/{len(SAMPLES)} samples ready")
    print("=" * 70)

    if success_count < len(SAMPLES):
        print()
        print("Some downloads failed. Please:")
        print("1. Check your internet connection")
        print("2. Verify ffmpeg is installed")
        print("3. Update placeholder URLs with real LibriVox/public domain URLs")
        return 1

    print()
    print("All samples downloaded successfully!")
    print(f"Audio files saved to: {AUDIO_DIR}/")
    print()
    print("Next steps:")
    print("  1. Run language detection: tenepal analyze validation_audio/eng_sample.wav")
    print("  2. Compare results to expected language")
    print("  3. Use analysis output to tune marker weights in Plan 02")
    return 0


if __name__ == "__main__":
    sys.exit(main())
