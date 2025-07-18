# -*- coding: utf-8 -*-
"""AIQry_Project.ipynb

Automatically generated by Colab.

file is located at
    https://colab.research.google.com/drive/1Dcr-ObdEMgEJ3sjK0YW5BqbTSAgu0p_j
"""

!pip install transformers torch gradio sentencepiece scikit-learn tensorflow pyod opencv-python mediapipe

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import cv2
import mediapipe as mp
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from pyod.models.auto_encoder import AutoEncoder

# ==============================================
# 1. Personalization & User Profiling
# ==============================================

class UserProfiler:
    def __init__(self):
        # Mock user data: [reading_speed, error_rate, preferred_font_size, color_sensitivity]
        self.user_data = np.array([
            [200, 0.15, 14, 0.7],  # User 1
            [180, 0.20, 16, 0.8],    # User 2
            [220, 0.10, 12, 0.6],    # User 3
            [190, 0.18, 15, 0.75],   # User 4
            [210, 0.12, 13, 0.65]    # User 5
        ])
        self.kmeans = KMeans(n_clusters=3)
        self.kmeans.fit(self.user_data)
        self.nn = NearestNeighbors(n_neighbors=2)
        self.nn.fit(self.user_data)

    def get_user_cluster(self, user_features):
        """Cluster users based on their reading patterns"""
        return self.kmeans.predict([user_features])[0]

    def recommend_settings(self, user_features):
        """Recommend settings based on similar users"""
        _, indices = self.nn.kneighbors([user_features])
        similar_users = self.user_data[indices[0]]
        return {
            'font_size': int(np.mean(similar_users[:, 2])),
            'color_scheme': 'dark' if np.mean(similar_users[:, 3]) > 0.7 else 'light',
            'reading_speed': int(np.mean(similar_users[:, 0]))
        }

# ==============================================
# 2. Text Processing & Feature Annotation
# ==============================================

class TextProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")
        self.pos_model = pipeline("token-classification",
                                model="vblagoje/bert-english-uncased-finetuned-pos")

    def annotate_text(self, text):
        """Annotate text with parts of speech and syllable information"""
        # Get POS tags
        pos_result = self.pos_model(text)

        # Simple syllable count approximation
        words = text.split()
        syllable_counts = [self._count_syllables(word) for word in words]

        # Phoneme highlighting (simplified)
        phonemes = [self._get_phonemes(word) for word in words]

        return {
            'pos_tags': pos_result,
            'syllable_counts': syllable_counts,
            'phonemes': phonemes
        }

    def _count_syllables(self, word):
        """Approximate syllable count"""
        vowels = "aeiouy"
        word = word.lower()
        count = 0

        if word[0] in vowels:
            count += 1

        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1

        if word.endswith("e"):
            count -= 1

        return max(count, 1)

    def _get_phonemes(self, word):
        """Simplified phoneme representation"""
        # In a real system, we'd use a proper phoneme dictionary
        phoneme_map = {
            'a': 'æ', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɛ',
            'f': 'f', 'g': 'g', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
            'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ɔ',
            'p': 'p', 'q': 'kw', 'r': 'r', 's': 's', 't': 't',
            'u': 'ʌ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j',
            'z': 'z'
        }
        return [phoneme_map.get(c.lower(), c) for c in word]

# ==============================================
# 3. Error Prediction & Correction Support
# ==============================================

from pyod.models.knn import KNN  # Simple alternative

class ErrorPredictor:
    def __init__(self):
        self.common_errors = {
            'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p',
            'm': 'w', 'w': 'm', 'n': 'u', 'u': 'n',
            'was': 'saw', 'on': 'no', 'from': 'form'
        }
        self.anomaly_detector = KNN(contamination=0.1)

    def _build_autoencoder(self):
        """Build a simple autoencoder for anomaly detection"""
        # Corrected PyOD AutoEncoder initialization
        model = AutoEncoder(
            epochs_num=50,  # Changed from 'epochs' to 'epochs_num'
            contamination=0.1,
            verbose=0
        )
        return model

    def predict_errors(self, word):
        """Predict likely dyslexic errors for a word"""
        similar = [v for k, v in self.common_errors.items() if k in word.lower()]
        return similar if similar else ["No common error patterns detected"]

    def detect_anomalies(self, reading_pattern):
        """Detect anomalies in reading patterns"""
        return self.autoencoder.fit_predict([reading_pattern])[0] == 1

# ==============================================
# 4. Gaze Tracking & Cognitive Load Estimation
# ==============================================

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Simple LSTM model for attention prediction
        self.attention_model = self._build_attention_model()

    def _build_attention_model(self):
        """Build a simple LSTM model for attention prediction"""
        model = Sequential([
            LSTM(64, input_shape=(10, 2)),  # 10 timesteps, 2 features (x,y)
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        # In a real system, we'd train this on actual gaze data
        return model

    def estimate_attention(self, gaze_points):
        """Estimate attention level from gaze points"""
        # Preprocess gaze points
        seq = np.array(gaze_points[-10:])  # Use last 10 points
        if len(seq) < 10:
            seq = np.pad(seq, ((0, 10-len(seq)), (0, 0)), 'constant')

        # Predict attention (1 = attentive, 0 = distracted)
        return self.attention_model.predict(np.expand_dims(seq, 0))[0][0] > 0.5

    def process_frame(self, frame):
        """Process a frame to detect gaze"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Get eye landmarks (simplified)
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = [landmarks[145], landmarks[159]]  # Example indices
            right_eye = [landmarks[374], landmarks[386]] # Example indices

            # Calculate simple gaze direction (center between eye points)
            gaze_x = (left_eye[0].x + right_eye[0].x) / 2
            gaze_y = (left_eye[0].y + right_eye[0].y) / 2

            return (gaze_x, gaze_y)
        return None

# ==============================================
# 5. Content Adaptation and Summarization
# ==============================================

class ContentAdapter:
    def __init__(self):
        # Load summarization models
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.simplifier_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.simplifier_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def simplify_text(self, text):
        """Simplify text using T5 model"""
        input_text = "simplify: " + text
        inputs = self.simplifier_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.simplifier_model.generate(
            inputs.input_ids,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        return self.simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize_text(self, text, summary_type="standard"):
        """Summarize text with options for different types"""
        if summary_type == "standard":
            result = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return result[0]['summary_text']
        elif summary_type == "simplified":
            simplified = self.simplify_text(text)
            result = self.summarizer(simplified, max_length=130, min_length=30, do_sample=False)
            return result[0]['summary_text']
        else:  # bullet points
            bullet_prompt = f"Summarize the following text in bullet points:\n\n{text}"
            result = self.summarizer(bullet_prompt, max_length=130, min_length=30, do_sample=False)
            return result[0]['summary_text']

# ==============================================
# Gradio Interface
# ==============================================

# Initialize components
user_profiler = UserProfiler()
text_processor = TextProcessor()
error_predictor = ErrorPredictor()
gaze_tracker = GazeTracker()
content_adapter = ContentAdapter()

def process_input(text, operation, webcam=None):
    """Process user input based on selected operation"""
    results = {}

    if operation == "user_profiling":
        # Mock user features: [reading_speed, error_rate, preferred_font_size, color_sensitivity]
        user_features = [200, 0.15, 14, 0.7]  # In real system, these would come from actual usage
        cluster = user_profiler.get_user_cluster(user_features)
        recommendations = user_profiler.recommend_settings(user_features)
        results = {
            "User Cluster": f"Cluster {cluster}",
            "Recommended Settings": recommendations
        }

    elif operation == "text_annotation":
        annotation = text_processor.annotate_text(text)
        results = {
            "POS Tags": "\n".join([f"{tag['word']}: {tag['entity']}" for tag in annotation['pos_tags']]),
            "Syllable Counts": "\n".join([f"{word}: {count}" for word, count in zip(text.split(), annotation['syllable_counts'])]),
            "Phonemes": "\n".join([f"{word}: {' '.join(phonemes)}" for word, phonemes in zip(text.split(), annotation['phonemes'])])
        }

    elif operation == "error_prediction":
        words = text.split()
        error_predictions = {}
        for word in words[:5]:  # Limit to first 5 words for demo
            errors = error_predictor.predict_errors(word)
            error_predictions[word] = errors
        results = {"Error Predictions": "\n".join([f"{k}: {', '.join(v)}" for k, v in error_predictions.items()])}

    elif operation == "gaze_tracking":
        if webcam is not None:
            # Process webcam frame
            frame = cv2.imread(webcam)
            gaze_point = gaze_tracker.process_frame(frame)

            # Mock gaze points for attention estimation
            gaze_points = [(0.5, 0.5) for _ in range(8)]  # First 8 points centered
            if gaze_point:
                gaze_points.append(gaze_point)

            attention = gaze_tracker.estimate_attention(gaze_points)
            results = {
                "Gaze Point": f"X: {gaze_point[0]:.2f}, Y: {gaze_point[1]:.2f}" if gaze_point else "Not detected",
                "Attention Level": "Attentive" if attention else "Distracted"
            }
        else:
            results = {"Error": "No webcam input provided"}

    elif operation == "content_adaptation":
        simplified = content_adapter.simplify_text(text)
        standard_summary = content_adapter.summarize_text(text, "standard")
        bullet_summary = content_adapter.summarize_text(text, "bullets")

        results = {
            "Simplified Text": simplified,
            "Standard Summary": standard_summary,
            "Bullet Point Summary": bullet_summary
        }

    return results

# Create Gradio interface
# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Dyslexia Support System")

    with gr.Tab("Main Interface"):
        with gr.Row():
            text_input = gr.Textbox(label="Input Text", lines=5)
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam (for gaze tracking)")

        operation = gr.Radio(
            choices=[
                "user_profiling",
                "text_annotation",
                "error_prediction",
                "gaze_tracking",
                "content_adaptation"
            ],
            label="Select Operation"
        )

        submit_btn = gr.Button("Process")

        output = gr.JSON(label="Results")

        submit_btn.click(
            fn=process_input,
            inputs=[text_input, operation, webcam_input],
            outputs=output
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## Dyslexia Support System

        This system implements five key ML components to support dyslexic readers:

        1. **Personalization & User Profiling**: Clusters users and recommends settings
        2. **Text Processing & Annotation**: Adds linguistic annotations to text
        3. **Error Prediction**: Identifies likely reading errors
        4. **Gaze Tracking**: Estimates attention using webcam
        5. **Content Adaptation**: Simplifies and summarizes text

        Note: This is a demo system with some simplified implementations.
        """)

demo.launch()
