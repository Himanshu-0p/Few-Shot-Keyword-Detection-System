import sys
import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, 
                            QLineEdit, QTextEdit, QGroupBox, QProgressBar)  # Added QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from scipy.signal import wiener

# Simplified Transformer Model for Keyword Detection
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x, mask=None):
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attended)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

class SimpleKWSModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(SimpleKWSModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.encoder = nn.ModuleList([
            TransformerEncoder(hidden_dim, hidden_dim*4, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, 1)
        
    def forward(self, audio_features):
        audio_features = audio_features.transpose(0, 1)
        audio_features = self.input_projection(audio_features)
        audio_features = self.positional_encoding(audio_features)
        
        encoded_audio = audio_features
        for layer in self.encoder:
            encoded_audio = layer(encoded_audio)
        
        output = self.output_projection(encoded_audio)
        return output.transpose(0, 1)

# Audio Processing
class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=512, hop_length=128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=None)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio
    
    def preprocess_audio(self, audio):
        filtered_audio = wiener(audio)
        normalized_audio = filtered_audio / (np.max(np.abs(filtered_audio)) + 1e-8)
        return normalized_audio
    
    def extract_features(self, audio):
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(stft)**2, sr=self.sample_rate, n_mels=self.n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

# Keyword Detector
class KeywordDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor()
        self.model = SimpleKWSModel()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def detect_keyword(self, audio_file, keyword, threshold=0.5):
        audio = self.processor.load_audio(audio_file)
        audio = self.processor.preprocess_audio(audio)
        features = self.processor.extract_features(audio)
        features_tensor = torch.tensor(features, dtype=torch.float32).T.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            scores = self.model(features_tensor)
            scores = torch.sigmoid(scores)
        
        max_score = torch.max(scores).item()
        if max_score > threshold:
            detected_frame = torch.argmax(scores).item()
            detected_time = detected_frame * self.processor.hop_length / self.processor.sample_rate
            return {
                'keyword': keyword,
                'timestamp': detected_time,
                'confidence': max_score
            }
        return None

# Processing Thread
class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    detection_result = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detector, audio_file, keyword, threshold):
        super().__init__()
        self.detector = detector
        self.audio_file = audio_file
        self.keyword = keyword
        self.threshold = threshold
        
    def run(self):
        try:
            for i in range(1, 101):
                self.update_progress.emit(i)
                time.sleep(0.02)
            result = self.detector.detect_keyword(self.audio_file, self.keyword, self.threshold)
            self.detection_result.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

# Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.detector = KeywordDetector(model_path="F:/code/FSLAKWS/models/fslakws_finetuned.pth")
        self.audio_path = None
        self.processing_thread = None
        self.player = QMediaPlayer()
        
    def init_ui(self):
        self.setWindowTitle("FSLAKWS - Keyword Spotting")
        self.setGeometry(100, 100, 600, 400)
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        header_label = QLabel("Keyword Spotting System")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        audio_group = QGroupBox("Audio Input")
        audio_layout = QVBoxLayout()
        self.audio_btn = QPushButton("Select Audio File")
        self.audio_btn.clicked.connect(self.select_audio)
        audio_layout.addWidget(self.audio_btn)
        self.audio_label = QLabel("No file selected")
        audio_layout.addWidget(self.audio_label)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        audio_layout.addWidget(self.play_btn)
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        keyword_group = QGroupBox("Keyword to Detect")
        keyword_layout = QVBoxLayout()
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Enter keyword (e.g., hello)")
        keyword_layout.addWidget(self.keyword_input)
        self.detect_btn = QPushButton("Detect Keyword")
        self.detect_btn.clicked.connect(self.detect_keyword)
        self.detect_btn.setEnabled(False)
        keyword_layout.addWidget(self.detect_btn)
        self.progress_bar = QProgressBar()
        keyword_layout.addWidget(self.progress_bar)
        keyword_group.setLayout(keyword_layout)
        main_layout.addWidget(keyword_group)
        
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def select_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")
        if file_path:
            self.audio_path = file_path
            self.audio_label.setText(os.path.basename(file_path))
            self.play_btn.setEnabled(True)
            self.check_detection_ready()
    
    def play_audio(self):
        if self.audio_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.audio_path)))
            self.player.play()
    
    def check_detection_ready(self):
        self.detect_btn.setEnabled(self.audio_path is not None and self.keyword_input.text().strip() != "")
    
    def detect_keyword(self):
        if not self.audio_path or not self.keyword_input.text().strip():
            return
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.detect_btn.setEnabled(False)
        keyword = self.keyword_input.text().strip()
        self.processing_thread = ProcessingThread(self.detector, self.audio_path, keyword, threshold=0.5)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.detection_result.connect(self.display_result)
        self.processing_thread.error_occurred.connect(self.display_error)
        self.processing_thread.finished.connect(lambda: self.detect_btn.setEnabled(True))
        self.processing_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def display_result(self, result):
        self.results_text.clear()
        if result is None:
            self.results_text.setText(f"Keyword '{self.keyword_input.text()}' not detected in the audio file.")
        else:
            self.results_text.setText(
                f"Keyword '{result['keyword']}' detected in the audio file at {result['timestamp']:.2f} seconds "
                f"with confidence {result['confidence']:.3f}"
            )
    
    def display_error(self, error):
        self.results_text.setText(f"Error: {error}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())