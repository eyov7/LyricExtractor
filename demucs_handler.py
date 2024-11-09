import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import tempfile
import os
import numpy as np
import librosa

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.sources = self.model.sources
        print(f"Model loaded successfully on {self.device}")
        print(f"Available sources: {self.sources}")
    def load_audio(self, file_path):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            print(f"Audio loaded - Shape: {waveform.shape}, Sample rate: {sample_rate}")
            
            # Handle mono input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            
            return waveform, sample_rate
        except Exception as e:
            print(f"Error loading with torchaudio: {e}")
            try:
                # Fallback to librosa
                audio, sr = librosa.load(file_path, sr=44100, mono=False)
                if audio.ndim == 1:
                    audio = np.vstack([audio, audio])
                waveform = torch.from_numpy(audio)
                return waveform, sr
            except Exception as e:
                raise RuntimeError(f"Failed to load audio: {str(e)}")

    def separate_vocals(self, audio_path):
        try:
            # Load audio
            waveform, sample_rate = self.load_audio(audio_path)
            print(f"Audio loaded - Shape: {waveform.shape}, Sample rate: {sample_rate}")
            
            # Ensure correct shape and device
            waveform = waveform.to(self.device)
            # Add batch dimension
            waveform = waveform.unsqueeze(0)
            
            # Process the entire audio at once instead of segments
            with torch.no_grad():
                sources = apply_model(self.model, waveform)
                
                # Get vocals
                vocals_idx = self.sources.index('vocals')
                vocals = sources[:, vocals_idx]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                torchaudio.save(
                    tmp.name,
                    vocals.squeeze(0).cpu(),
                    sample_rate,
                    format='wav'
                )
                return tmp.name
                
        except Exception as e:
            raise RuntimeError(f"Separation failed: {str(e)}")

def configure_model():
    return {
        "segment_size": 8 if torch.cuda.is_available() else 4,
        "overlap": 0.1,
        "sample_rate": 44100,
        "channels": 2
    }

def check_dependencies():
    try:
        import torch
        import torchaudio
        import librosa
        import demucs
        from demucs.pretrained import get_model
        
        # Test audio loading
        test_audio = np.random.random(44100)
        test_tensor = torch.from_numpy(test_audio)
        
        print("All required packages are installed correctly")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        return False
