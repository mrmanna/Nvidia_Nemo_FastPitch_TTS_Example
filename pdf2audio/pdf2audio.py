import pdfplumber
import torch
import nemo.collections.tts as nemo_tts
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import argparse
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import os
import re

# Set environment variable for memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load models once
def load_models():
    # Load the FastPitch model
    fastpitch_model = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker").to(device).eval()
    # Load the HiFi-GAN model
    hifigan_model = HifiGanModel.from_pretrained("tts_en_hifitts_hifigan_ft_fastpitch").to(device).eval()
    return fastpitch_model, hifigan_model

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle None if text extraction fails
    return text

# Function to preprocess text (remove invalid characters, symbols)
def preprocess_text(text):
    # Remove unwanted characters and symbols using regex
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)  # Keep only letters, numbers, and common punctuation
    return text

# Function to convert text to speech
def text_to_speech(text,fastpitch_model, hifigan_model, output_file="output.wav"):
       
    with torch.no_grad():  # Disable gradient calculations
        # Convert text to spectrogram
        parsed = fastpitch_model.parse(text)

#speaker id:
#    92     Cori Samuel
#    6097   Phil Benson
#    9017   John Van Stan
#    6670   Mike Pelton
#    6671   Tony Oliva
#    8051   Maria Kasper
#    9136   Helen Taylor
#    11614  Sylviamb
#    11697  Celine Major
#    12787  LikeManyWaters

        spectrogram = fastpitch_model.generate_spectrogram(tokens=parsed,speaker=92)
        # Convert spectrogram to audio
        audio = hifigan_model.convert_spectrogram_to_audio(spec=spectrogram)
        
    # Ensure audio is in the correct format
    audio = audio.cpu().numpy()
    
    # Debugging: Print statistics about the audio
   # print(f"Audio min: {np.min(audio)}, max: {np.max(audio)}, dtype: {audio.dtype}")
    
    # Normalize audio to ensure it is within [-1.0, 1.0]
    audio = np.clip(audio, -1.0, 1.0)
    
    # Convert to int16 format
    audio = np.int16(audio * 32767)
    
    # Debugging: Print statistics after conversion
   # print(f"Audio after scaling shape: {audio.shape},shape1: {audio.shape[1]}, min: {np.min(audio)}, max: {np.max(audio)}, dtype: {audio.dtype}")
    
    # # Ensure audio data is 1D
    # if len(audio.shape) > 1:
    #     audio = audio.flatten()
    if len(audio.shape) > 1:
        if audio.shape[0] == 1:  # Mono
            audio = audio[0]  # Convert from (1, N) to (N,)
        elif audio.shape[1] == 2:  # Stereo
            # Stereo handling, if applicable
            audio = audio.astype(np.int16)  # Ensure the audio is in the correct format
        else:
            raise ValueError("Unsupported audio channel format")
    else:
        # If audio has no channel dimension, treat it as mono
        audio = audio.flatten()
    
    # # Verify that audio is within valid range for int16
    if np.any(audio < -32768) or np.any(audio > 32767):
        raise ValueError("Audio data out of bounds for int16 format")
    
    # Save audio
    try:
        write(output_file, 44100, audio)  # Save audio with scipy
    except ValueError as e:
        print(f"Error writing audio file: {e}")
        print(f"Audio data range: min={np.min(audio)}, max={np.max(audio)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# Function to split text into chunks by sentence
def split_text_by_sentence(text, max_chunk_size=250):
    # Split text by sentence-ending punctuation marks (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding the next sentence would exceed the chunk size
        if len(current_chunk) + len(sentence) > max_chunk_size:
            # Save the current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Otherwise, keep adding sentences to the current chunk
            current_chunk += " " + sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to process text in chunks
def process_text_in_chunks(text, fastpitch_model, hifigan_model, output_base_path, chunk_size=250):
    chunk_files = []
    # Use the updated split_text_by_sentence function to create chunks without splitting sentences
    chunks = split_text_by_sentence(text, chunk_size)
    
    for i, chunk in enumerate(chunks):
        chunk_file = f"{output_base_path}_chunk_{i}.wav"
        # Call the text_to_speech function for each chunk
        text_to_speech(chunk, fastpitch_model, hifigan_model, output_file=chunk_file)
        chunk_files.append(chunk_file)
    
    return chunk_files
def merge_audio_files(file_list, output_file):
    # Load all audio chunks and concatenate them
    combined = AudioSegment.from_wav(file_list[0])
    for file in file_list[1:]:
        audio = AudioSegment.from_wav(file)
        combined += audio
    
    # Export the combined audio to a single file
    combined.export(output_file, format="wav")
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a PDF book to an audiobook.")
    parser.add_argument("pdf_path", type=str, help="Path to the source PDF file.")
    parser.add_argument("output_wav_path", type=str, help="Base path to save the output WAV files.")
       # Parse arguments
    args = parser.parse_args()
      # Load the models once
    fastpitch_model, hifigan_model = load_models()
        # Extract text from the PDF
    text = extract_text_from_pdf(args.pdf_path)
    # Process text in chunks
    chunk_files = process_text_in_chunks(text, fastpitch_model, hifigan_model,output_base_path=args.output_wav_path)
    
    # Merge all audio chunks into one file
    merge_audio_files(chunk_files, output_file=args.output_wav_path)
    
    # Optionally clean up chunk files
    for chunk_file in chunk_files:
        os.remove(chunk_file)

if __name__ == "__main__":
    main()
