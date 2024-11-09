from transformers import pipeline
import torch
import torchaudio

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            device=self.device
        )
    
    def transcribe(self, audio_path):
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Convert to numpy array
        audio_array = waveform.squeeze().numpy()
        
        # Transcribe to text
        result = self.model(
            audio_array,
            **self.process_options()
        )
        
        # Extract and format transcription
        transcription = result["text"]
        
        return transcription

    def process_options(self):
        return {
            'language': 'en',
            'task': 'transcribe',
            'return_timestamps': True,
            'chunk_length_s': 30,
            'stride_length_s': 5
        }