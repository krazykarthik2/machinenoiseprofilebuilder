import streamlit as st
import os
import tempfile
import zipfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from utils import (
    extract_audio_from_video,
    load_audio,
    extract_features_chunked,
    save_features,
    train_machine_model,
    predict_machine,
    DATA_DIR
)

st.set_page_config(page_title="Audio Noise Profile Builder (Unsupervised)", layout="wide")

st.title("Audio Noise Profile Builder")
st.write("Build an unsupervised profile for a specific machine to distinguish it from normal day-to-day noise.")

# Sidebar for Machine Management
st.sidebar.header("Machine Profiles")
existing_machines = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

new_machine = st.sidebar.text_input("Create new machine profile:")
if st.sidebar.button("Add Machine"):
    if new_machine and new_machine not in existing_machines:
        os.makedirs(os.path.join(DATA_DIR, new_machine), exist_ok=True)
        st.sidebar.success(f"Added {new_machine}")
        st.rerun()

if not existing_machines and not new_machine:
    st.warning("Please create a machine profile in the sidebar first.")
    st.stop()
    
active_machine = st.sidebar.selectbox("Select Active Machine for Data Collection", [""] + existing_machines)

st.header("1. Upload Data to Build Profile")
st.write(f"Upload audio/video files for the machine **{active_machine if active_machine else '[Select a machine]'}**.")
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
    features_2d = extract_features_chunked(y, sr)
    return features_2d, y, sr, audio_path

if uploaded_files:
    all_features_chunks = []
    first_y, first_sr, first_audio_path = None, None, None
    
    with st.spinner("Chunking audio and extracting features..."):
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
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
                                all_features_chunks.append(feats)
                                if first_y is None:
                                    first_y, first_sr, first_audio_path = y, sr, a_path
                            except Exception as e:
                                st.warning(f"Could not process {file}: {e}")
            else:
                try:
                    feats, y, sr, a_path = process_file(tfile.name, file_ext)
                    all_features_chunks.append(feats)
                    if first_y is None:
                        first_y, first_sr, first_audio_path = y, sr, a_path
                except Exception as e:
                    st.warning(f"Could not process {uploaded_file.name}: {e}")
                    
    if all_features_chunks:
        # Stack all chunks vertically into one large 2D dataset representing the machine's full distribution
        aggregated_dataset = np.vstack(all_features_chunks)
        st.success(f"Extracted {len(aggregated_dataset)} chunks of features from the uploaded files.")

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
                st.write("Mel-Spectrogram")
                S = librosa.feature.melspectrogram(y=first_y, sr=first_sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 3))
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=first_sr, ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                st.pyplot(fig)

        if active_machine:
            if st.button(f"Save Profile for '{active_machine}'"):
                save_features(active_machine, aggregated_dataset)
                st.success(f"Successfully saved {len(aggregated_dataset)} feature chunks as a single profile to {active_machine}!")
        else:
            st.warning("Select an active machine in the sidebar to save this profile.")

st.header("2. Model Training (Unsupervised)")
st.write("Train a One-Class SVM on a machine's profile. This model learns ONLY the features of this machine and rejects anything else (like normal room noise).")

if st.button("Train Unsupervised Model"):
    if active_machine:
        with st.spinner(f"Training One-Class SVM for {active_machine}..."):
            success, msg = train_machine_model(active_machine)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.error("Please select an Active Machine in the sidebar first.")

st.header("3. Run Inference on Live Audio")
st.write("Upload an audio clip or record from your microphone. The system will check if any of the trained machines are detected.")

col1, col2 = st.columns(2)
with col1:
    test_file = st.file_uploader("Upload Audio/Video", type=['wav', 'mp3', 'mp4', 'mov', 'avi', 'mpeg'], key="test")
with col2:
    # Use native Streamlit audio input if available
    mic_audio = None
    if hasattr(st, 'audio_input'):
        mic_audio = st.audio_input("Record from Microphone")
    else:
        st.info("Update Streamlit to v1.38+ for native microphone support.")

audio_source = mic_audio if mic_audio else test_file

if audio_source is not None:
    file_ext = audio_source.name.split('.')[-1].lower() if hasattr(audio_source, 'name') else 'wav'
    
    with st.spinner("Analyzing and predicting..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") 
        tfile.write(audio_source.read())
        tfile.close()
        
        try:
            if file_ext in ['mp4', 'mov', 'avi', 'mpeg']:
                temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                audio_path = extract_audio_from_video(tfile.name, temp_audio_path)
            else:
                audio_path = tfile.name
                
            y, sr = load_audio(audio_path)
            
            prediction, match_scores, chunk_preds = predict_machine(y, sr)
            
            if prediction == "Unknown / Background Noise":
                st.warning(f"**Result: {prediction}**")
                st.write("The audio does not match any of our trained machine profiles.")
            else:
                st.success(f"**Result: Detected Machine '{prediction}'!**")
                
            st.write("Match Ratios (Inlier Percentage):")
            st.json(match_scores)
            
            st.subheader("Chunk-by-Chunk Analysis (Confidence Gradient)")
            st.write("Flowing gradient based on model confidence. Deep green = strong match, Deep red = strong outlier.")
            fig, ax = plt.subplots(figsize=(12, 3))
            
            if len(chunk_preds) > 0:
                import matplotlib.colors as mcolors
                import matplotlib.cm as cm
                
                # Setup colormap normalization based on the decision function scores
                # 0 is the boundary (yellow), >0 is inlier (green), <0 is outlier (red)
                vmin = min(-1.0, np.min(chunk_preds))
                vmax = max(1.0, np.max(chunk_preds))
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
                # Stretch the scores to fit behind the waveform using a bicubic gradient
                gradient = np.array(chunk_preds).reshape(1, -1)
                
                # librosa waveshow limits
                times = librosa.times_like(y, sr=sr)
                max_time = times[-1] if len(times) > 0 else len(chunk_preds)
                
                # Plot the flowing gradient background
                ax.imshow(gradient, aspect='auto', cmap='RdYlGn', norm=norm,
                          extent=[0, max_time, -1, 1],
                          alpha=0.4, interpolation='bicubic')
            
            librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.8, color='black')
            ax.set_ylim([-1, 1])
                
            st.pyplot(fig)
            st.audio(audio_path)
            
        except Exception as e:
            st.error(f"Error analyzing file: {e}")
