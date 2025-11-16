import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QWidget, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# --- Reused from predict.py ---
MODEL_PATH = "./models/emotion_classifier"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class EmotionAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("Emotion-Aware Analyzer")
        self.setGeometry(300, 300, 600, 400)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Title
        title_label = QLabel("Emotion-Aware NLP Analyzer")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Please enter an English sentence, and the model will analyze the emotions it contains.")
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Input Box
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Enter English text here...")
        self.input_text.setFont(QFont("Arial", 12))
        layout.addWidget(self.input_text)

        # Analyze Button
        self.analyze_btn = QPushButton("Analyze Emotions")
        self.analyze_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)  # Disabled until model is loaded
        layout.addWidget(self.analyze_btn)

        # Log/Output Box
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Arial", 11))
        layout.addWidget(self.log_text)

        # Status Label
        self.status_label = QLabel("Status: Loading model...")
        self.status_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.status_label)

    def load_model(self):
        """Load the model and tokenizer."""
        try:
            self.log_text.append(f"Device: {DEVICE}")
            self.log_text.append(f"Loading model and tokenizer from {MODEL_PATH}...")
            QApplication.processEvents()  # Refresh the UI

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
            self.model.eval()  # Set to evaluation mode

            self.log_text.append("✅ Model loaded successfully! Please enter text.")
            self.status_label.setText("Status: Model Ready")
            self.analyze_btn.setEnabled(True)

        except EnvironmentError:
            self.log_text.append(f"❌ Error: Model not found at {MODEL_PATH}.")
            self.log_text.append("Please run 'python src/train.py' first to train and save the model.")
            self.status_label.setText("Status: Model Load Failed")
            self.analyze_btn.setEnabled(False)

    def start_analysis(self):
        text = self.input_text.text().strip()  # Use .strip() to remove whitespace
        if not text:
            self.log_text.append("\n❌ Error: Input text cannot be empty.")
            return

        self.log_text.append(f"\n--- Analyzing ---")
        self.log_text.append(f"Input: {text}")

        try:
            # Reuse logic from predict_emotions
            # 1. Prepare inputs
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

            # 2. Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 3. Process outputs
            logits = outputs.logits
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze())

            # 4. Get predictions based on threshold (using 0.3 as default for GUI)
            threshold = 0.3
            predictions = (probs > threshold).int().cpu().numpy()

            # 5. Convert to emotion names
            detected_emotions = []
            for i, label_id in enumerate(predictions):
                if label_id == 1:
                    detected_emotions.append(LABELS[i])

            if not detected_emotions:
                detected_emotions = ["neutral"]  # If no emotions detected, return "neutral"

            self.log_text.append(f"Detected Emotions: {detected_emotions}")

            # (Optional) Show more detailed probabilities
            self.log_text.append("\nDetailed Probabilities (Top 5):")
            top_probs, top_indices = torch.topk(probs, 5)
            for i in range(5):
                label_name = LABELS[top_indices[i].item()]
                prob = top_probs[i].item()
                if prob > 0.1:  # Only show if confidence is above a small threshold
                    self.log_text.append(f"  - {label_name}: {prob:.2%}")

        except Exception as e:
            self.log_text.append(f"\n❌ Error during analysis: {e}")


def main():
    app = QApplication(sys.argv)
    window = EmotionAnalyzerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
