# Audio Lyrics Pipeline: Vocal Isolation and Transcription

## Project Objective
Create an ML pipeline that processes audio files through vocal isolation and lyrics transcription, with a Gradio interface for deployment on Hugging Face Spaces.

## Technical Requirements

### Dependencies
```bash
pip install gradio>=4.0.0
pip install demucs>=4.0.0
pip install transformers>=4.30.0
pip install torch>=2.0.0 torchaudio>=2.0.0
```

### File Structure
```
project/
├── app.py
├── demucs_handler.py
├── whisper_handler.py
└── requirements.txt
```

## Implementation Details

### demucs_handler.py
```python
import torch
import torchaudio
from demucs import pretrained

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.get_model(model_name)
        self.model.to(self.device)
    
    def separate_vocals(self, audio_path):
        # Load audio
        # Process stems
        # Return vocals only
        pass

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
```

### whisper_handler.py
```python
from transformers import pipeline
import torch

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
        # Transcribe to text
        # Return transcription
        pass

    def process_options(self):
        return {
            'language': 'en',
            'task': 'transcribe',
            'return_timestamps': True,
            'chunk_length_s': 30,
            'stride_length_s': 5
        }
```

### app.py
```python
import gradio as gr
from demucs_handler import DemucsProcessor
from whisper_handler import WhisperTranscriber

def create_interface():
    processor = DemucsProcessor()
    transcriber = WhisperTranscriber()
    
    def process_audio(audio_file, whisper_model="base"):
        # Process audio through pipeline
        # Return results
        pass

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(type="filepath"),
            gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="base",
                label="Whisper Model Size"
            )
        ],
        outputs=[
            gr.Audio(label="Isolated Vocals"),
            gr.Textbox(label="Transcribed Lyrics")
        ],
        title="Audio Lyrics Extractor",
        description="Upload audio to extract vocals and transcribe lyrics"
    )
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
```

## Model Specifications

### Demucs
- Model: htdemucs (default)
- Input: Stereo audio at 44.1kHz
- Output: Separated vocals
- Memory: 3GB+ GPU recommended
- CPU fallback available

### Whisper
- Models available:
  - tiny: 39M parameters
  - base: 74M parameters
  - small: 244M parameters
  - medium: 769M parameters
  - large-v3: 1.55B parameters
- Input: Mono audio at 16kHz
- Languages: 99 languages supported
- Memory requirements vary by model size

## Error Handling
- Audio format validation
- GPU memory management
- Temporary file cleanup
- Progress tracking
- Graceful fallbacks

## Performance Considerations
- Demucs processes in stereo (2 channels)
- Whisper expects mono input
- Sample rate conversion needed
- Memory optimization for large files
- Batch processing capabilities
- Caching for repeated operations

## Notes
- Ensure proper GPU drivers for CUDA support
- Monitor memory usage during processing
- Consider implementing caching for repeated operations
- Add logging for debugging


