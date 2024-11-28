import streamlit as st
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
import tempfile
import os
from scipy.stats import entropy
from scipy.signal import find_peaks
import lightgbm as lgb
import pickle
import matplotlib.pyplot as plt
import librosa.display
import parselmouth
import lime
import lime.lime_tabular

def extract_features(y, sr):
    features = {}
    
    # Ensure minimum duration
    min_duration = 0.1  # 100 ms
    if len(y) / sr < min_duration:
        y = librosa.util.fix_length(y, size=int(min_duration * sr))
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    features.update({f'mfcc_{i}': np.mean(mfcc) for i, mfcc in enumerate(mfccs)})
    
    # Delta and Delta-Delta MFCCs
    if mfccs.shape[1] >= 3:
        delta_mfccs = librosa.feature.delta(mfccs, mode='nearest')
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, mode='nearest')
        features.update({f'delta_mfcc_{i}': np.mean(delta_mfcc) for i, delta_mfcc in enumerate(delta_mfccs)})
        features.update({f'delta2_mfcc_{i}': np.mean(delta2_mfcc) for i, delta2_mfcc in enumerate(delta2_mfccs)})
    else:
        features.update({f'delta_mfcc_{i}': 0 for i in range(13)})
        features.update({f'delta2_mfcc_{i}': 0 for i in range(13)})
    
    # Pitch features
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) > 0:
        features['pitch_mean'] = np.mean(pitches)
        features['pitch_std'] = np.std(pitches)
        pitch_periods = 1 / pitches
        features['ppe'] = entropy(pitch_periods)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['ppe'] = 0
    
    # Voice quality features
    try:
        sound = parselmouth.Sound(y, sr)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        
        features['jitter'] = jitter
        features['shimmer'] = shimmer
        features['hnr_mean'] = np.mean(hnr.values)
    except:
        features['jitter'] = 0
        features['shimmer'] = 0
        features['hnr_mean'] = 0
    
    # Temporal features
    zero_crossings = librosa.zero_crossings(y)
    features['speech_rate'] = sum(zero_crossings) / len(y) * sr
    
    # Pause characteristics
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    features['pause_count'] = max(0, len(non_silent_intervals) - 1)
    if features['pause_count'] > 0:
        features['pause_duration_mean'] = np.mean([interval[0] - non_silent_intervals[i-1][1] 
                                                 for i, interval in enumerate(non_silent_intervals[1:], 1)]) / sr
    else:
        features['pause_duration_mean'] = 0
    
    # Energy-based features
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return pd.DataFrame([features])

def extract_features_with_visualization(y, sr):
    features = extract_features(y, sr)
    
    st.subheader("Audio Visualizations")
    
    # Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # Plot MFCCs
    plt.figure(figsize=(10, 4))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # Mel Spectrogram
    plt.figure(figsize=(10, 4))
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # RMS Energy
    plt.figure(figsize=(10, 4))
    rms = librosa.feature.rms(y=y)
    times = librosa.times_like(rms)
    plt.plot(times, rms[0])
    plt.title('RMS Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # Zero Crossing Rate
    plt.figure(figsize=(10, 4))
    zcr = librosa.feature.zero_crossing_rate(y)
    plt.plot(times, zcr[0])
    plt.title('Zero Crossing Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('ZCR')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    return features

def get_lime_explanation(model_data, features):
    # Create a prediction function that handles scaling internally
    def predict_fn(x):
        scaled_x = model_data['scaler'].transform(x)
        return model_data['model'].predict_proba(scaled_x)
    
    # Create the explainer using the current features as training data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=model_data['scaler'].transform(features),
        feature_names=features.columns,
        class_names=['No Dysarthria', 'Dysarthria'],
        mode='classification',
        discretize_continuous=True
    )
    
    # Generate explanation
    exp = explainer.explain_instance(
        features.iloc[0],
        predict_fn,
        num_features=15
    )
    #print the raw lime explanation
    print(exp.as_list())
    
    fig = exp.as_pyplot_figure()
    return fig

def load_model():
    with open('LGBM_CPU_final.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict(features, model_data):
    scaled_features = model_data['scaler'].transform(features)
    prediction = model_data['model'].predict(scaled_features)
    probability = model_data['model'].predict_proba(scaled_features)
    return prediction, probability

def process_audio_with_visualization(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = extract_features_with_visualization(y, sr)
    return features

def display_results(prediction, probability, features, model_data):
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("🔴 Dysarthria Detected")
            prob_value = probability[0][1]
        else:
            st.success("🟢 No Dysarthria Detected")
            prob_value = probability[0][0]
    
    with col2:
        st.metric("Confidence", f"{prob_value:.2%}")
    
    st.write("LIME Explanation (Local Feature Importance)")
    lime_fig = get_lime_explanation(model_data, features)
    st.pyplot(lime_fig)
    plt.close()

def main():
    st.set_page_config(
        page_title="Dysarthria Detection System",
        page_icon="🎤",
        layout="wide"
    )

    st.title("🎤 Dysarthria Detection System")
    
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            st.audio(uploaded_file)
            if st.button("Analyze Uploaded Audio"):
                with st.spinner("Extracting features and analyzing..."):
                    features = process_audio_with_visualization(audio_path)
                    model_data = load_model()
                    prediction, probability = predict(features, model_data)
                    display_results(prediction, probability, features, model_data)
                os.unlink(audio_path)
    
    with tab2:
        st.write("Click the button below to start recording")
        audio_bytes = audio_recorder()
        
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                audio_path = tmp_file.name
            
            st.audio(audio_bytes)
            if st.button("Analyze Recorded Audio"):
                with st.spinner("Extracting features and analyzing..."):
                    features = process_audio_with_visualization(audio_path)
                    model_data = load_model()
                    prediction, probability = predict(features, model_data)
                    display_results(prediction, probability, features, model_data)
                os.unlink(audio_path)

if __name__ == "__main__":
    main()
