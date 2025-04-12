import streamlit as st
import noisereduce as nr
import librosa
import soundfile as sf
import os
import tempfile

st.set_page_config(page_title="Noise Reducer", layout="centered")
st.title("🔉 Noise Reducer App")
st.markdown("Upload a `.wav` or `.mp3` audio file and get a cleaned version!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

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

        # Save cleaned file
        cleaned_filename = uploaded_file.name.replace(".", "_cleaned.")
        cleaned_path = os.path.join(temp_dir, cleaned_filename)
        sf.write(cleaned_path, reduced_noise, sr)

    st.success("Done! 🎉 Your cleaned file is ready.")
    with open(cleaned_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Cleaned Audio",
            data=f,
            file_name=cleaned_filename,
            mime="audio/wav"
        )
