import torch
from transformers import pipeline
import soundfile as sf
import numpy as np
            from scipy.signal import resample

class WhisperTranscriber:
    def __init__(self, model_size="large-v2"):
        self.model_size = model_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize pipeline with memory optimizations
        self.model = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            chunk_length_s=45,           # Optimal for musical phrases or long segments
            device=self.device,
            batch_size=8,                # Will adjust based on GPU memory
            torch_dtype=torch.float16,   # Use half precision
            return_timestamps=True
        )
    
    def preprocess_audio(self, audio_path):
        # Load audio with soundfile for better memory efficiency
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        # Ensure 16-bit PCM format
        audio_data = (audio_data * 32767).astype('int16')
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            audio_data = resample(audio_data, int(16000 / sample_rate * len(audio_data)))
            sample_rate = 16000
        
        return audio_data, sample_rate
    
    def transcribe(self, audio_path):
        try:
            # Preprocess audio
            audio_data, sample_rate = self.preprocess_audio(audio_path)
            print(f"Audio loaded and preprocessed - Shape: {audio_data.shape}, Sample rate: {sample_rate}")
            
            # Transcribe
            result = self.model(
                audio_data,
                sampling_rate=sample_rate,
                generate_kwargs={
                    "task": "transcribe",
                    "max_new_tokens": 256
                }
            )

            # Extract transcription
            transcription = result["text"] if isinstance(result, dict) else result
            return transcription
            
        except Exception as e:
            print(f"Error in transcribe: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    transcriber = WhisperTranscriber(model_size="large-v2")
    transcription = transcriber.transcribe("path_to_your_audio_file.wav")
    print(f"Transcription: {transcription}")
