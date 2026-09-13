import os
import numpy as np
import librosa
import joblib
from moviepy import VideoFileClip
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features_single(y, sr):
    features = []
    
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))
    
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bw))
    features.append(np.std(bw))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))
    
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for e in mfccs:
        features.append(np.mean(e))
        features.append(np.std(e))
        
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    features.append(np.mean(tempogram))
    features.append(np.std(tempogram))
    
    return np.array(features)

def extract_features_chunked(y, sr, chunk_duration=1.0):
    chunk_samples = int(chunk_duration * sr)
    features_list = []
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        if len(chunk) < chunk_samples / 2:
            continue
        feats = extract_features_single(chunk, sr)
        features_list.append(feats)
    if len(features_list) == 0:
        return np.array([extract_features_single(y, sr)])
    return np.array(features_list)

def save_features(machine_name, features_2d):
    machine_dir = os.path.join(DATA_DIR, machine_name)
    os.makedirs(machine_dir, exist_ok=True)
    
    feature_path = os.path.join(machine_dir, "profile.npy")
    np.save(feature_path, features_2d)
    return feature_path

def train_machine_model(machine_name):
    machine_dir = os.path.join(DATA_DIR, machine_name)
    feature_path = os.path.join(machine_dir, "profile.npy")
    
    if not os.path.exists(feature_path):
        return False, f"No profile found for {machine_name}."
        
    X = np.load(feature_path)
    if len(X) == 0:
        return False, "Profile is empty."
        
    # Filter out transient states (startup/shutdown/trailing bits) 
    # by assuming the "main" machine noise is the most dense cluster.
    # We remove the 20% of chunks that are furthest from the center.
    if len(X) > 10:
        temp_scaler = StandardScaler()
        X_temp_scaled = temp_scaler.fit_transform(X)
        centroid = np.mean(X_temp_scaled, axis=0)
        distances = np.linalg.norm(X_temp_scaled - centroid, axis=1)
        
        # Keep the 80% closest to the centroid (the steady-state noise)
        threshold = np.percentile(distances, 80)
        X = X[distances <= threshold]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Unsupervised learning: One-Class SVM learns the boundaries of this specific machine's noise
    # nu=0.05 because we already aggressively filtered outliers
    clf = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
    clf.fit(X_scaled)
    
    joblib.dump(scaler, os.path.join(machine_dir, "scaler.pkl"))
    joblib.dump(clf, os.path.join(machine_dir, "model.pkl"))
    
    return True, f"Unsupervised model trained successfully for {machine_name}."

def predict_machine(y, sr):
    X_test = extract_features_chunked(y, sr)
    machines = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    best_machine = None
    highest_inlier_ratio = 0.0
    
    results = {}
    
    for machine in machines:
        machine_dir = os.path.join(DATA_DIR, machine)
        scaler_path = os.path.join(machine_dir, "scaler.pkl")
        model_path = os.path.join(machine_dir, "model.pkl")
        
        if os.path.exists(scaler_path) and os.path.exists(model_path):
            scaler = joblib.load(scaler_path)
            clf = joblib.load(model_path)
            
            X_scaled = scaler.transform(X_test)
            preds = clf.predict(X_scaled)
            
            # preds is 1 for inlier, -1 for outlier
            inliers = np.sum(preds == 1)
            ratio = inliers / len(preds)
            results[machine] = ratio
            
            if ratio > highest_inlier_ratio and ratio > 0.5: # At least 50% of chunks must match the machine profile
                highest_inlier_ratio = ratio
                best_machine = machine
                
    if best_machine is None:
        return "Unknown / Background Noise", results
        
    return best_machine, results
