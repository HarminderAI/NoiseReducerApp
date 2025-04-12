import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import os
import tempfile
from pydub.utils import which
from pydub import AudioSegment
import numpy as np

# Set the path to ffmpeg for pydub
AudioSegment.converter = which("ffmpeg")

st.set_page_config(page_title="Noise Reducer", layout="centered")
st.title("🔉 Noise Reducer App")
st.markdown("Upload a `.wav` or `.mp3` audio file and get a cleaned version!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

def limit_audio(audio_segment, threshold=-1.0):
    """
    Limits audio volume to prevent clipping.
    """
    return audio_segment.apply_gain(threshold - audio_segment.dBFS)

def normalize_audio(audio_segment):
    """
    Normalize audio to ensure consistent volume levels.
    """
    return audio_segment.normalize()

if uploaded_file is not None:
    with st.spinner("Processing... Please wait ⏳"):
        # Save to a temporary location
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load audio
        y, sr = librosa.load(input_path, sr=None)

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=y, sr=sr)

        # Save cleaned file temporarily to apply Pydub normalization
        temp_cleaned_path = os.path.join(temp_dir, "temp_cleaned.wav")
        sf.write(temp_cleaned_path, reduced_noise, sr)

        # Load the cleaned audio file with Pydub
        audio = AudioSegment.from_wav(temp_cleaned_path)

        # Apply noise reduction if further needed
        reduced_audio_segment = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )

        # Normalize the cleaned audio
        normalized_audio = normalize_audio(reduced_audio_segment)

        # Limit audio volume to avoid clipping
        limited_audio = limit_audio(normalized_audio)

        # Save the cleaned and volume-limited audio
        cleaned_filename = uploaded_file.name.replace(".", "_cleaned.")
        cleaned_path = os.path.join(temp_dir, cleaned_filename)
        limited_audio.export(cleaned_path, format="wav")

        st.success("Done! 🎉 Your cleaned file is ready.")

    # Play cleaned audio before download
    st.audio(cleaned_path, format="audio/wav")

    with open(cleaned_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Cleaned Audio",
            data=f,
            file_name=cleaned_filename,
            mime="audio/wav"
        )
