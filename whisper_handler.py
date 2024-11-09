from transformers import pipeline
import torch
import torchaudio
import numpy as np

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
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Initial waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1 and waveform.shape > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                print(f"After mono conversion shape: {waveform.shape}")
            
            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                print(f"After resampling shape: {waveform.shape}")
            
            # Ensure we're working with the right shape
            audio_data = waveform.squeeze().numpy()
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=0)
            print(f"Audio data shape before chunking: {audio_data.shape}")
            
            # Chunk audio into 30-second segments
            chunk_length_samples = 30 * 16000  # 30 seconds at 16kHz
            
            # Process each chunk with overlap
            transcriptions = []
            start_idx = 0
            
            while start_idx < len(audio_data):
                end_idx = min(start_idx + chunk_length_samples, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Pad last chunk if necessary
                if len(chunk) < chunk_length_samples:
                    chunk = np.pad(chunk, (0, chunk_length_samples - len(chunk)))
                
                print(f"Processing chunk {len(transcriptions) + 1}, length: {len(chunk)}")
                # Transcribe chunk
                result = self.model(
                    chunk,
                    **self.process_options()
                )
                
                transcription = result["text"].strip()
                if transcription:  # Only add non-empty transcriptions
                    transcriptions.append(transcription)
                
                # Move to next chunk with overlap
                start_idx += chunk_length_samples // 2  # 50% overlap
            
            # Combine transcriptions
            full_transcription = ' '.join(transcriptions)
            print(f"Transcription complete, found {len(transcriptions)} chunks")
            
            return full_transcription
            
        except Exception as e:
            print(f"Error in transcribe: {str(e)}")
            raise

    def process_options(self):
        return {
            'max_new_tokens': 2048,  # Increased for longer transcriptions
            'chunk_length_s': 30,    # Default chunk length
            'stride_length_s': 15,   # Overlap for better continuity
            'return_timestamps': False
        }
