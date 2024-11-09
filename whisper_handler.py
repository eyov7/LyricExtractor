import torch
from transformers import pipeline
import librosa
import soundfile as sf
import numpy as np
class WhisperTranscriber:
    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            chunk_length_s=30,
            device=self.device,
            batch_size=8,
            torch_dtype=torch.float16,
            return_timestamps=True
            )

    def preprocess_audio(self, audio_path, target_sr=16000):
        # Load audio with librosa for better preprocessing
        y, sr = librosa.load(audio_path, sr=None)
        
        # Resample to 16kHz (Whisper's expected rate)
        y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        
        # Apply noise reduction
        y_cleaned = librosa.effects.preemphasis(y_resampled)
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_cleaned)
        
        # Remove silence and very quiet parts
        y_filtered = librosa.effects.trim(
            y_normalized,
            top_db=30,
            frame_length=2048,
            hop_length=512
        )[0]
        
        return y_filtered, target_sr
    
    def transcribe(self, audio_path):
        try:
            # Preprocess audio
            audio_data, sample_rate = self.preprocess_audio(audio_path)
            print(f"Audio loaded and preprocessed - Shape: {audio_data.shape}, Sample rate: {sample_rate}")
            
            # Transcribe
            result = self.model(
                audio_data,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "en",
                    "max_new_tokens": 256,
                    "temperature": 0.7  # Added to reduce hallucination
                }
            )

            # Extract transcription with timestamps if available
            if isinstance(result, dict):
                if "chunks" in result:
                    transcription = " ".join([chunk["text"] for chunk in result["chunks"]])
                else:
                    transcription = result["text"]
            else:
                transcription = result
                
            return transcription
            
        except Exception as e:
            print(f"Error in transcribe: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    transcriber = WhisperTranscriber(model_size="medium")
    transcription = transcriber.transcribe("path_to_your_audio_file.wav")
    print(f"Transcription: {transcription}")
