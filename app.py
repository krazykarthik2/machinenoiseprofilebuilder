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
    audio_path = file_path
    if file_ext in ['mp4', 'mov', 'avi', 'mpeg']:
        try:
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            audio_path = extract_audio_from_video(file_path, temp_audio_path)
        except Exception as e:
            # If moviepy fails (e.g. unsupported codec or audio-only), fallback to directly loading with librosa
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
            
            st.subheader("Interactive Chunk-by-Chunk Analysis")
            st.write("Click anywhere on the waveform to play the audio at that exact moment! The flowing background gradient shows the model's confidence (Deep Green = Machine, Deep Red = Outlier).")
            
            if len(chunk_preds) > 0:
                import matplotlib.colors as mcolors
                import matplotlib.cm as cm
                import base64
                import streamlit.components.v1 as components
                
                # Setup colormap normalization based on the decision function scores
                vmin = min(-1.0, np.min(chunk_preds))
                vmax = max(1.0, np.max(chunk_preds))
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                cmap = cm.get_cmap('RdYlGn')
                
                # Build CSS linear gradient string
                stops = []
                n = len(chunk_preds)
                for i, score in enumerate(chunk_preds):
                    rgba = cmap(norm(score))
                    # Convert to rgba string for CSS with some transparency
                    rgba_str = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.6)"
                    pct = (i / max(1, n-1)) * 100
                    stops.append(f"{rgba_str} {pct:.1f}%")
                
                css_gradient = f"linear-gradient(to right, {', '.join(stops)})"
                
                # Encode audio to base64
                with open(audio_path, 'rb') as f:
                    audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                    
                # Create Wavesurfer HTML
                html_code = f"""
                <div style="background: {css_gradient}; border-radius: 8px; padding: 10px; box-shadow: inset 0 0 10px rgba(0,0,0,0.1);">
                    <div id="waveform"></div>
                </div>
                <div style="margin-top: 15px; text-align: center;">
                    <button id="playBtn" style="padding: 10px 24px; font-size: 16px; font-weight: bold; cursor: pointer; border-radius: 8px; border: none; background: #2e3b4e; color: white; transition: 0.2s;">▶ Play / Pause</button>
                </div>
                <script type="module">
                    import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js'
                    
                    const ws = WaveSurfer.create({{
                        container: '#waveform',
                        waveColor: 'rgba(0, 0, 0, 0.4)',
                        progressColor: 'rgba(0, 0, 0, 0.8)',
                        url: 'data:audio/wav;base64,{audio_b64}',
                        height: 120,
                        normalize: true,
                        cursorColor: '#ff0000',
                        cursorWidth: 2
                    }})
                    
                    const btn = document.getElementById('playBtn')
                    btn.onclick = () => ws.playPause()
                    
                    ws.on('play', () => btn.textContent = '⏸ Pause')
                    ws.on('pause', () => btn.textContent = '▶ Play / Pause')
                </script>
                """
                
                components.html(html_code, height=250)
            else:
                st.audio(audio_path)
            
        except Exception as e:
            st.error(f"Error analyzing file: {e}")

st.header("4. Global 3D Cluster Visualizer")
st.write("Visualize all collected 1-second chunks across all saved machines in 3D. We use PCA (Principal Component Analysis) to project the features down to 3 dimensions linearly. This preserves the natural geometric structure without warping space, so you can see exactly how the machines form distinct natural clusters.")

if st.button("Generate 3D Cluster Map"):
    import plotly.express as px
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    all_chunks = []
    chunk_labels = []
    
    machines = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for machine in machines:
        profile_path = os.path.join(DATA_DIR, machine, "profile.npy")
        if os.path.exists(profile_path):
            X_mach = np.load(profile_path)
            if len(X_mach) > 0:
                all_chunks.extend(X_mach)
                chunk_labels.extend([machine] * len(X_mach))
                
    if len(all_chunks) < 3:
        st.error("Not enough data to create a 3D plot. Please build profiles for your machines first.")
    else:
        with st.spinner("Projecting chunks to 3D space..."):
            # Standardize
            X_scaled = StandardScaler().fit_transform(all_chunks)
            
            # Linear projection to 3 components (no non-linear warping)
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'PC3': X_pca[:, 2],
                'Machine': chunk_labels
            })
            
            # Create interactive 3D scatter plot
            fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
                                color='Machine',
                                title="3D Machine Noise Clusters",
                                opacity=0.7)
            
            # Make markers a bit smaller and cleanly styled
            fig.update_traces(marker=dict(size=4, line=dict(width=0)))
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3'
            ))
            
            st.plotly_chart(fig, use_container_width=True)
