import os
import json
import numpy as np
import librosa
import joblib
from moviepy.editor import VideoFileClip
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr):
    features = []
    
    # Noise profile characteristics
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))
    
    # Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bw))
    features.append(np.std(bw))
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for e in mfccs:
        features.append(np.mean(e))
        features.append(np.std(e))
        
    # Rhythmic/Beat features (Tempogram for machine repetition)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    features.append(np.mean(tempogram))
    features.append(np.std(tempogram))
    
    return np.array(features)

def save_features(machine_name, features):
    machine_dir = os.path.join(DATA_DIR, machine_name)
    os.makedirs(machine_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(machine_dir) if f.endswith('.npy')]
    feature_path = os.path.join(machine_dir, f"features_{len(existing_files)}.npy")
    np.save(feature_path, features)
    return feature_path

def train_model():
    X = []
    y_labels = []
    
    machines = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(machines) < 2:
        return False, "Need at least two machines to train."
        
    for machine in machines:
        machine_dir = os.path.join(DATA_DIR, machine)
        for f in os.listdir(machine_dir):
            if f.endswith('.npy'):
                feats = np.load(os.path.join(machine_dir, f))
                X.append(feats)
                y_labels.append(machine)
                
    if len(X) < 2:
        return False, "Not enough data points."
        
    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y_labels)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(list(set(y_labels)), os.path.join(MODELS_DIR, "classes.pkl"))
    
    return True, "Model trained successfully."

def predict_machine(features):
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None, "Model not trained yet."
        
    scaler = joblib.load(scaler_path)
    clf = joblib.load(model_path)
    
    features_scaled = scaler.transform([features])
    prediction = clf.predict(features_scaled)[0]
    probabilities = clf.predict_proba(features_scaled)[0]
    
    classes = clf.classes_
    prob_dict = {c: p for c, p in zip(classes, probabilities)}
    
    return prediction, prob_dict
