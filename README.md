# Machine Audio Noise Profile Builder

An advanced, interactive Streamlit application designed to build, analyze, and visualize unsupervised acoustic profiles for various mechanical machines. 

By extracting high-resolution spectral features at 0.1-second intervals, this tool learns the exact acoustic fingerprint—including rhythmic oscillations and timbres—of a machine's steady state, effectively ignoring human speech and background anomalies.

## 🚀 Key Features

*   **Unsupervised Anomaly Detection:** Uses a One-Class Support Vector Machine (SVM) to learn the "normal" continuous state of a specific machine without needing negative examples.
*   **High-Resolution Feature Extraction:** Uses `librosa` to extract 34 spectral features (including MFCCs, Spectral Centroid, Bandwidth, Rolloff, and Zero-Crossing Rate) every 100ms.
*   **Bulk Zipped Imports:** Need to map 10 machines? Just drag and drop `.zip` files (e.g., `Fan.zip`, `Drill.zip`) containing audio/video into the sidebar to automatically process and train the models instantly.
*   **Interactive Inference Visualizer:** Test new audio or record live via your microphone. The app generates an interactive `Wavesurfer.js` waveform colored with a flowing confidence gradient. Green chunks perfectly match the machine's profile, while Red chunks highlight outliers (like someone talking).
*   **Global 3D Topological Visualizer:** Project the high-dimensional acoustic features of all your machines into an interactive 3D space using **UMAP**. This preserves complex oscillating topologies (manifolds) so you can visually confirm how different machines separate.
*   **Natural Density Clustering:** Toggle on **DBSCAN** within the 3D visualizer to group chunks purely based on mathematical density, allowing you to see if your manual machine labels align with the natural mathematical clusters.

## 🛠️ Installation & Quickstart

1.  **Clone the repository**
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The app uses `librosa` and `moviepy` which can handle most standard audio/video formats directly.*
3.  **Run the application:**
    ```bash
    python -m streamlit run app.py
    ```

## 🧠 How the Algorithm Works

1.  **Slicing:** Audio is chopped into precise 0.1-second chunks.
2.  **Mapping:** 34 distinct spectral features are extracted for each chunk to capture both tone and dynamic range.
3.  **Filtering:** The system calculates the mathematical centroid of the data and explicitly drops the top 20% most distant chunks to filter out startup/shutdown transient noises.
4.  **Enclosing:** A One-Class SVM with an RBF kernel draws a tight boundary around the dense core of the remaining machine chunks. Anything that falls outside this strict boundary during inference is instantly flagged as an anomaly.

## 📦 How Bulk Import Works (Step-by-Step)

1. **Upload Zips:** You drag and drop one or multiple `.zip` files (e.g., `Generator.zip`, `HVAC.zip`) into the Bulk Import sidebar.
2. **Auto-Naming:** The app reads the filename (e.g., `Generator`) and automatically registers a new machine profile under that exact name.
3. **Extraction & Scanning:** It extracts the zip into a temporary directory and recursively scans all internal folders looking for valid media files (`.wav`, `.mp3`, `.mp4`, `.mpeg`, etc.).
4. **Processing:** Every valid audio/video file it finds is silently run through the feature extraction pipeline (chopped into 0.1s chunks and mapped to 34 features).
5. **Aggregation:** The features from *all* the files inside the zip are aggregated and stacked into a single massive dataset (`profile.npy`).
6. **Auto-Training:** The app instantly trains the One-Class SVM on this aggregated dataset. By the time the loading spinner finishes, your machine is fully trained and immediately ready for live inference and 3D visualization!
