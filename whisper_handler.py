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
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        # Chunk audio into 30-second segments
        chunk_length = 30 * 16000  # 30 seconds at 16kHz
        audio_data = waveform.squeeze().numpy()
        
        # Calculate number of samples per chunk and total chunks
        samples_per_chunk = int(chunk_length)
        total_chunks = int(np.ceil(len(audio_data) / samples_per_chunk))
        
        # Process each chunk with overlap
        transcriptions = []
        for i in range(total_chunks):
            start_idx = i * samples_per_chunk
            end_idx = min(start_idx + samples_per_chunk, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
        
            # Pad last chunk if necessary
            if len(chunk) < samples_per_chunk:
                chunk = np.pad(chunk, (0, samples_per_chunk - len(chunk)))
            # Transcribe chunk
            result = self.model(
                chunk,
                **self.process_options()
            )

            transcription = result["text"].strip()
            if transcription:  # Only add non-empty transcriptions
                transcriptions.append(transcription)
        
        # Combine transcriptions
        full_transcription = ' '.join(transcriptions)
        
        return full_transcription

    def process_options(self):
        return {
            'max_new_tokens': 2048,  # Increased for longer transcriptions
            'chunk_length_s': 30,    # Default chunk length
            'stride_length_s': 15,   # Overlap for better continuity
            'return_timestamps': False
        }
