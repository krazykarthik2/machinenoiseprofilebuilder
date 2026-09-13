import streamlit as st
import os
import tempfile
import zipfile
import uuid
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
st.write("Build simple noise profiles for machine sounds and train a lightweight model to recognize them.")

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
uploaded_files = st.file_uploader(
    "Upload Audio/Video files or a ZIP file", 
    type=['wav', 'mp3', 'mp4', 'mov', 'avi', 'mpeg', 'zip'],
    accept_multiple_files=True
)

def process_file(file_path, file_ext):
    if file_ext in ['mp4', 'mov', 'avi', 'mpeg']:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio_path = extract_audio_from_video(file_path, temp_audio_path)
    else:
        audio_path = file_path
        
    y, sr = load_audio(audio_path)
    features = extract_features(y, sr)
    return features, y, sr, audio_path

if uploaded_files:
    all_features = []
    first_y, first_sr, first_audio_path = None, None, None
    
    with st.spinner("Processing files..."):
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") 
            tfile.write(uploaded_file.read())
            tfile.close()
            
            if file_ext == 'zip':
                st.info(f"Extracting {uploaded_file.name}...")
                extract_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(tfile.name, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        ext = file.split('.')[-1].lower()
                        if ext in ['wav', 'mp3', 'mp4', 'mov', 'avi', 'mpeg']:
                            extracted_file_path = os.path.join(root, file)
                            try:
                                feats, y, sr, a_path = process_file(extracted_file_path, ext)
                                all_features.append(feats)
                                if first_y is None:
                                    first_y, first_sr, first_audio_path = y, sr, a_path
                            except Exception as e:
                                st.warning(f"Could not process {file}: {e}")
            else:
                try:
                    feats, y, sr, a_path = process_file(tfile.name, file_ext)
                    all_features.append(feats)
                    if first_y is None:
                        first_y, first_sr, first_audio_path = y, sr, a_path
                except Exception as e:
                    st.warning(f"Could not process {uploaded_file.name}: {e}")
                    
    st.success(f"Successfully extracted features from {len(all_features)} files.")

    if first_y is not None:
        st.subheader("Sample Noise Profile Visualization")
        st.write("Displaying visualization for the first processed file.")
        st.audio(first_audio_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Waveform")
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(first_y, sr=first_sr, ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("Mel-Spectrogram (Frequency Domain)")
            S = librosa.feature.melspectrogram(y=first_y, sr=first_sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(10, 3))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=first_sr, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
            
        st.write("Rhythmic Analysis (Tempogram)")
        onset_env = librosa.onset.onset_strength(y=first_y, sr=first_sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=first_sr)
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(tempogram, sr=first_sr, x_axis='time', y_axis='tempo', ax=ax)
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)

    if active_machine and all_features:
        aggregated_profile = np.mean(all_features, axis=0)
        if st.button(f"Save Profile for '{active_machine}'"):
            save_features(active_machine, aggregated_profile)
            st.success(f"Successfully aggregated {len(all_features)} files and saved as a single profile to {active_machine}!")
    elif not active_machine:
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
test_file = st.file_uploader("Upload single Audio/Video to Identify Machine", type=['wav', 'mp3', 'mp4', 'mov', 'avi', 'mpeg'], key="test")

if test_file is not None:
    file_ext = test_file.name.split('.')[-1].lower()
    
    with st.spinner("Analyzing and predicting..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") 
        tfile.write(test_file.read())
        tfile.close()
        
        try:
            features, _, _, _ = process_file(tfile.name, file_ext)
            prediction, probs = predict_machine(features)
            
            if prediction is None:
                st.error(probs) 
            else:
                st.success(f"**Predicted Machine: {prediction}**")
                st.write("Confidence:")
                st.json(probs)
        except Exception as e:
            st.error(f"Error analyzing file: {e}")
