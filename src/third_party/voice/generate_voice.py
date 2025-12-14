import requests
from bs4 import BeautifulSoup
import os
from pydub import AudioSegment
import argparse # New import
import pydub
from datasets import load_dataset
import torch
from TTS.api import TTS
import numpy as np
import soundfile as sf 

## 1. ArticleScraper: Fetching and Cleaning Content
class ArticleScraper:
    """Scrapes a URL and cleans the HTML content to extract readable text."""
    
    def __init__(self, url: str):
        self.url = url
        self.raw_text = None

    def scrape(self):
        """Fetches the URL and extracts all paragraph text."""
        print(f"Scraping content from: {self.url}")
        try:
            response = requests.get(self.url, timeout=15)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # A simple cleaning strategy: extract text from <p> tags
            paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
            
            # Join paragraphs, ensuring two newlines for good separation
            self.raw_text = '\n\n'.join(filter(None, paragraphs))
            
            print(f"Scraping complete. Extracted {len(self.raw_text)} characters.")
            return self.raw_text
            
        except requests.exceptions.RequestException as e:
            print(f"Error during scraping: {e}")
            return None

## 2. AudioSampler: Reference Audio Management
class AudioSampler:
    """Handles downloading and managing the reference audio file for TTS."""
    
    def __init__(self, save_path: str = "celebrity_voice_sample.wav"):
        self.save_path = save_path
        
    def download_sample(self):
            # Load dataset using the stable version (assuming datasets==2.16.1 is installed)
            dataset = load_dataset("sdialog/voices-celebrities", split="train")
            
            # Access the first sample's audio dictionary
            first_sample = dataset[1]
            audio_data_dict = first_sample['audio'] 
            audio_array = audio_data_dict['array']
            sampling_rate = audio_data_dict['sampling_rate']
            
            # Ensure the array is a standard floating-point format expected by soundfile
            if audio_array.dtype != np.float32 and audio_array.dtype != np.float64:
                audio_array = audio_array.astype(np.float32)

            # 2. Save the audio array to a standard WAV file
            sf.write(self.save_path, audio_array, sampling_rate)
            print(f"\n Audio successfully saved to: {self.save_path}")

    
## 3. TTSProcessor: Text Chunking and Audio Generation
class TTSProcessor:
    """Manages the TTS model and handles chunking text for generation."""
    
    def __init__(self, model_path: str = "tts_model/xtts_v2"):
        # Class 2 in your prompt is now integrated here (running the TTS model)
        self.model = TTS(model_path, progress_bar=False).to(device)
        print(f"model_path {model_path}")
        self.output_dir = "tts_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_files = []

    def _chunk_text(self, text: str, max_chars: int = 500) -> list[str]:
        """Splits the article text into chunks based on sentence boundaries and max length."""
        
        # A simple chunking strategy using split on newlines
        sentences = text.split('\n\n') 
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding the next sentence exceeds the max length, save the current chunk
            if len(current_chunk) + len(sentence) + 2 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "\n\n"
            else:
                current_chunk += sentence + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        print(f"Article split into {len(chunks)} chunks.")
        return chunks

    def process_article(self, article_text: str, ref_audio_path: str):
        """Chunks the text, runs TTS on each chunk, and saves the file paths."""
        if not article_text:
            print("No article text to process.")
            return
            
        text_chunks = self._chunk_text(article_text)
        
        for i, chunk in enumerate(text_chunks):
            output_file = os.path.join(self.output_dir, f"chunk_{i:03d}.mp3")
            print(chunk)
            
            # Calls the TTS model to generate audio for this chunk
            self.model.tts_to_file(
                text=chunk, 
                ref_audio_path=ref_audio_path, 
                output_file=output_file,
                language="en"
            )
            self.audio_files.append(output_file)
            
        print(f"TTS generation complete for all {len(self.audio_files)} chunks.")
        return self.audio_files

## 4. AudioJoiner: Combining all generated audio files
class AudioJoiner:
    """Collects all chunked audio files and merges them into a single track."""
    
    def __init__(self, chunk_files: list[str], output_filename: str = "final_article_audio.mp3"):
        # The list of file paths collected by the TTSProcessor class
        self.chunk_files = chunk_files
        self.output_filename = output_filename

    def combine_audio(self):
        """Combines all audio files using pydub."""
        if not self.chunk_files:
            print("❌ Error: No audio chunks found to combine.")
            return None
            
        print(f"\nMerging {len(self.chunk_files)} audio chunks into one file...")
        
        # 1. Initialize an empty audio segment
        final_audio = AudioSegment.empty()
        
        # 2. Iterate and append each chunk
        for filepath in self.chunk_files:
            if not os.path.exists(filepath):
                print(f"⚠️ Warning: File not found: {filepath}. Skipping.")
                continue

            # Load the audio chunk using pydub. 
            # pydub automatically detects the file format (wav, mp3, etc.)
            segment = AudioSegment.from_file(filepath)
            
            # Append the loaded segment to the growing final audio track
            final_audio += segment
                
         

if __name__ == "__main__":
    # --- Configuration from Command Line ---
    ARTICLE_URL = "https://www.cnn.com/2025/12/12/health/fda-black-box-warning-covid-vaccine"
    REF_AUDIO_PATH = "celebrity_voice_sample.wav" # Remains a static save name, but you could make this an argument too
    model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
    # model_path = "tts_models/de/thorsten/tacotron2-DDC"
    # model_path = "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"
    # 1. Scrape the Article
    scraper = ArticleScraper(ARTICLE_URL)
    article_text = scraper.scrape()

    if not article_text:
        print("Pipeline aborted due to scraping failure.")
    else:
        # 2. Download Sample Audio
        sampler = AudioSampler(REF_AUDIO_PATH)
        sampler.download_sample()

        # 3. Process Text with TTS Model
        # Pass the model_path from the arguments
        processor = TTSProcessor(model_path) 
        chunk_files = processor.process_article(article_text, REF_AUDIO_PATH)

        # 4. Join All Audio Chunks
        joiner = AudioJoiner(chunk_files)
        final_file = joiner.combine_audio()

        print("\nPipeline finished.")
