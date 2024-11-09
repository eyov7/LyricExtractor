import gradio as gr
from demucs_handler import DemucsProcessor
from whisper_handler import WhisperTranscriber
import os

def create_interface():
    processor = DemucsProcessor()
    transcriber = WhisperTranscriber()
    
    def process_audio(audio_file, whisper_model="base"):
        try:
            # Isolate vocals
            vocals_path = processor.separate_vocals(audio_file)
            
            # Transcribe lyrics
            if whisper_model != transcriber.model_size:
                transcriber.__init__(whisper_model)
            lyrics = transcriber.transcribe(vocals_path)
            
            return vocals_path, lyrics
        except Exception as e:
            return None, f"Error: {str(e)}"
        finally:
            # Clean up temporary files
            if os.path.exists(vocals_path):
                os.remove(vocals_path)

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(type="filepath", label="Input Audio"),
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
