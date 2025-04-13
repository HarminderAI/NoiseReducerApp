import streamlit as st 
import noisereduce as nr
import librosa
import librosa.display
import soundfile as sf
import os
import tempfile
import warnings
import re
import matplotlib.pyplot as plt
from pydub.utils import which
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import scipy.signal

# Suppress specific warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set the path to ffmpeg for pydub
AudioSegment.converter = which("ffmpeg")

# Streamlit UI
st.set_page_config(page_title="Noise Reducer", layout="centered")
st.info("✅ App loaded successfully!")
st.title("🔉 Noise Reducer App (Optimized for Lectures)")
st.markdown("Upload a `.wav` or `.mp3` lecture and get a **cleaned, clearer version**!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# Advanced controls
st.markdown("### 🎛️ Optional Controls")
noise_strength = st.slider("Noise Reduction Strength", 0.0, 1.0, 0.3, step=0.05)
clarity_boost_db = st.slider("Clarity Boost (Post-cleaning Volume Gain)", 0, 10, 2, step=1)
apply_advanced = st.checkbox("🧠 Apply High-Pass Filter (Removes fan/rumble)", value=True)
apply_silence_trim = st.checkbox("✂️ Trim Long Pauses", value=False)
output_format = st.radio("Choose Output Format", ["WAV", "MP3"])

# Helpers
def high_pass_filter(y, sr, cutoff=100):
    b, a = scipy.signal.butter(1, cutoff / (0.5 * sr), btype='high', analog=False)
    return scipy.signal.filtfilt(b, a, y)

def limit_audio(audio_segment, threshold=-1.0):
    return audio_segment.apply_gain(threshold - audio_segment.dBFS)

def normalize_audio(audio_segment):
    return audio_segment.normalize()

def plot_waveform(y, sr, title):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

MAX_FILE_SIZE_MB = 50

if uploaded_file is not None:
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.warning(f"⚠️ File is too large. Please upload a file smaller than {MAX_FILE_SIZE_MB} MB.")
            st.stop()

        with st.spinner("Processing... Please wait ⏳"):
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name

            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            st.markdown(f"**Sample Rate:** {sr} Hz  |  **Duration:** {duration:.2f} seconds")

            # Waveform: Original
            st.markdown("#### 🔍 Original Waveform")
            st.pyplot(plot_waveform(y, sr, "Original Audio"))

            # Step 1: High-pass filter
            if apply_advanced:
                y = high_pass_filter(y, sr)

            # Step 2: Noise reduction
            reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=noise_strength)

            # Save cleaned temp WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_cleaned:
                sf.write(tmp_cleaned.name, reduced_noise, sr)
                cleaned_path = tmp_cleaned.name

            # Load with Pydub
            audio = AudioSegment.from_wav(cleaned_path)

            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Normalize
            normalized_audio = normalize_audio(audio)

            # Limit volume
            limited_audio = limit_audio(normalized_audio)

            # Clarity boost
            enhanced_audio = limited_audio + clarity_boost_db

            # Trim silence if selected
            if apply_silence_trim:
                chunks = split_on_silence(
                    enhanced_audio,
                    min_silence_len=1000,
                    silence_thresh=enhanced_audio.dBFS - 14,
                    keep_silence=500
                )
                enhanced_audio = sum(chunks)

            # Final output file
            file_ext = "mp3" if output_format == "MP3" else "wav"
            
            # Sanitize uploaded filename
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", uploaded_file.name)
            cleaned_filename = safe_name.replace(".", f"_cleaned.")
            cleaned_path_final = cleaned_filename.rsplit(".", 1)[0] + "." + file_ext
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_out:
                enhanced_audio.export(tmp_out.name, format=file_ext)
                final_output_path = tmp_out.name

            # Waveform: Cleaned
            y_cleaned, _ = librosa.load(final_output_path, sr=sr)
            st.markdown("#### ✨ Cleaned Waveform")
            st.pyplot(plot_waveform(y_cleaned, sr, "Cleaned Audio"))

            st.success("✅ Done! Your cleaned lecture audio is ready.")

            # Audio Comparison
            st.markdown("### 🎧 Comparison: Original vs Cleaned Audio")
            st.audio(input_path, format="audio/wav")
            st.markdown("**Original Audio**")
            st.audio(final_output_path, format=f"audio/{file_ext}")
            st.markdown("**Cleaned Audio**")

            # Download
            with open(final_output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Cleaned Audio",
                    data=f,
                    file_name=cleaned_path_final,
                    mime=f"audio/{'mpeg' if file_ext == 'mp3' else 'wav'}"
                )
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
