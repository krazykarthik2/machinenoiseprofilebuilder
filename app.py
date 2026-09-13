import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from utils import (
    extract_audio_from_video,
    load_audio,
    extract_features,
    save_features,
    train_model,
    predict_machine,
    DATA_DIR
)

st.set_page_config(page_title="Audio Noise Profile Builder", layout="wide")

st.title("Audio Noise Profile Builder")
st.write("Build simple noise profiles for machine sounds (e.g., white, pink, rhythmic noise) and train a lightweight model to recognize them.")

# Sidebar for Machine Management
st.sidebar.header("Machine Profiles")
existing_machines = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

new_machine = st.sidebar.text_input("Create new machine:")
if st.sidebar.button("Add Machine"):
    if new_machine and new_machine not in existing_machines:
        os.makedirs(os.path.join(DATA_DIR, new_machine), exist_ok=True)
        st.sidebar.success(f"Added {new_machine}")
        st.rerun()

if not existing_machines and not new_machine:
    st.warning("Please create a machine profile in the sidebar first.")
    st.stop()
    
active_machine = st.sidebar.selectbox("Select Active Machine for Data Collection", [""] + existing_machines)

st.header("1. Upload and Process Data")
uploaded_file = st.file_uploader("Upload Audio/Video of the Machine", type=['wav', 'mp3', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Processing file..."):
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") 
        tfile.write(uploaded_file.read())
        tfile.close()
        
        audio_path = tfile.name
        
        if file_ext in ['mp4', 'mov', 'avi']:
            st.info("Extracting audio from video...")
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio_path = extract_audio_from_video(tfile.name, temp_audio_path)
            
        y, sr = load_audio(audio_path)
        
        st.audio(audio_path)
        
        # Plotting
        st.subheader("Noise Profile Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Waveform")
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("Mel-Spectrogram (Frequency Domain)")
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(10, 3))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
            
        st.write("Rhythmic Analysis (Tempogram)")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(tempogram, sr=sr, x_axis='time', y_axis='tempo', ax=ax)
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

        # Extract Features
        st.subheader("Feature Extraction")
        features = extract_features(y, sr)
        st.write(f"Extracted a feature vector of length {len(features)}. These features represent the machine's distinct acoustic footprint (spectral and rhythmic properties).")
        
        if active_machine:
            if st.button(f"Save Profile for '{active_machine}'"):
                save_features(active_machine, features)
                st.success(f"Features saved to {active_machine}!")
        else:
            st.warning("Select an active machine in the sidebar to save this profile.")

st.header("2. Model Training")
if st.button("Train Recognition Model"):
    with st.spinner("Training model on collected profiles..."):
        success, msg = train_model()
        if success:
            st.success(msg)
        else:
            st.error(msg)
            
st.header("3. Machine Recognition (Inference)")
test_file = st.file_uploader("Upload Audio/Video to Identify Machine", type=['wav', 'mp3', 'mp4', 'mov', 'avi'], key="test")

if test_file is not None:
    file_ext = test_file.name.split('.')[-1].lower()
    
    with st.spinner("Analyzing and predicting..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") 
        tfile.write(test_file.read())
        tfile.close()
        
        audio_path = tfile.name
        
        if file_ext in ['mp4', 'mov', 'avi']:
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio_path = extract_audio_from_video(tfile.name, temp_audio_path)
            
        y, sr = load_audio(audio_path)
        features = extract_features(y, sr)
        
        prediction, probs = predict_machine(features)
        
        if prediction is None:
            st.error(probs) # error message
        else:
            st.success(f"**Predicted Machine: {prediction}**")
            st.write("Confidence:")
            st.json(probs)
