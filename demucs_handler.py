import torch
import torchaudio
from demucs import pretrained
import os

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.get_model(model_name)
        self.model.to(self.device)
    
    def separate_vocals(self, audio_path):
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure stereo audio at 44.1kHz
        if waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)
        if sample_rate != 44100:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
        
        # Process stems
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        
        with torch.no_grad():
            sources = self.model.separate(waveform.to(self.device))
            sources = sources.cpu()
        
        # Extract vocals
        vocals = sources[self.model.sources.index('vocals')]
        
        # Normalize and convert to int16
        vocals = vocals / vocals.abs().max()
        vocals = (vocals * 32767).to(torch.int16)
        
        # Save vocals to temporary file
        temp_path = 'temp_vocals.wav'
        torchaudio.save(temp_path, vocals, 44100)
        
        return temp_path

    def configure_processing(self):
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        if gpu_memory < 3e9:  # Less than 3GB
            return {"device": "cpu"}
        elif gpu_memory < 7e9:  # Less than 7GB
            return {
                "device": "cuda",
                "segment_size": 8,
                "overlap": 0.1
            }
        return {"device": "cuda"}

    def __del__(self):
        # Clean up temporary file
        if os.path.exists('temp_vocals.wav'):
            os.remove('temp_vocals.wav')
