import os
import sys
import logging
import gradio as gr
import shutil
from demucs_handler import DemucsProcessor, check_dependencies, configure_model
from whisper_handler import WhisperTranscriber
import tempfile
import torch
import torchaudio
import soundfile as sf
import librosa
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_environment():
    try:
        import torch
        import torchaudio
        import demucs
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"Torchaudio version: {torchaudio.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logging.error(f"Environment validation failed: {e}")
        sys.exit(1)

def create_interface():
    validate_environment()
    processor = DemucsProcessor()
    transcriber = WhisperTranscriber()
    
    def process_audio(audio_file, whisper_model="base", progress=gr.Progress()):
        if audio_file is None:
            return None, "Please upload an audio file."
        
        temp_files = []
        try:
            progress(0, desc="Starting processing")
            logging.info(f"Processing file: {audio_file}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_audio_path = os.path.join(temp_dir, "input.wav")
                vocals_output_path = os.path.join(temp_dir, "vocals.wav")
                
                # Convert to WAV first
                audio, sr = librosa.load(audio_file, sr=44100)
                # Fixed: use samplerate instead of sr
                sf.write(temp_audio_path, audio, samplerate=sr)
                temp_files.append(temp_audio_path)
                
                progress(0.1, desc="Separating vocals")
                try:
                    vocals_path = processor.separate_vocals(temp_audio_path)
                    # Copy vocals to output path
                    shutil.copy2(vocals_path, vocals_output_path)
                    temp_files.append(vocals_output_path)
                except RuntimeError as e:
                    logging.error(f"Vocal separation failed: {str(e)}")
                    return None, f"Vocal separation failed: {str(e)}"
                
                # Load the processed vocals for playback
                vocals_audio, vocals_sr = librosa.load(vocals_output_path, sr=None)
                
                progress(0.75, desc="Transcribing")
                lyrics = transcriber.transcribe(vocals_output_path)
                progress(1.0, desc="Processing complete")
                
                # Return the audio data tuple and lyrics
                return (vocals_sr, vocals_audio), lyrics
                
        except Exception as e:
            error_message = f"Processing error: {str(e)}"
            logging.error(error_message)
            return None, error_message
        finally:
            # Cleanup temporary files
            for file in temp_files:
                if file and os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(label="Upload Audio File", type="filepath"),
            gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v2"],
                value="medium",
                label="Whisper Model Size"
            )
        ],
        outputs=[
            gr.Audio(label="Isolated Vocals", type="numpy"),
            gr.Textbox(label="Transcribed Lyrics", lines=10, max_lines=20)
        ],
        title="Audio Lyrics Extractor",
        description="Upload an audio file to extract vocals and transcribe lyrics",
        analytics_enabled=False
    )
    return interface

if __name__ == "__main__":
    if not check_dependencies():
        print("Please install missing dependencies")
        exit(1)
    interface = create_interface()
    interface.launch()
