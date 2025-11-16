import torch
import argparse
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data_loader import LABELS

# Path where the trained model is saved
MODEL_PATH = "./models/emotion_classifier"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
try:
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")
except EnvironmentError:
    print(f"Error: Model not found at {MODEL_PATH}.", file=sys.stderr)
    print("Please run 'python src/train.py' first to train and save the model.", file=sys.stderr)
    sys.exit(1)


def predict_emotions(text: str, threshold: float = 0.3) -> list:
    """
    Predicts emotions for a given input text.

    Args:
        text (str): The input text from the user.
        threshold (float): Probability threshold for classifying an emotion (0.3-0.5 is reasonable).

    Returns:
        list: A list of detected emotion labels.
    """
    # 1. Prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    # 2. Model inference
    with torch.no_grad():  # Disable gradient calculation to save memory and speed up
        outputs = model(**inputs)

    # 3. Process outputs
    logits = outputs.logits

    # 4. Apply Sigmoid
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze())  # (28,)

    # 5. Get predictions based on threshold
    predictions = (probs > threshold).int().cpu().numpy()

    # 6. Convert to emotion names
    detected_emotions = []
    for i, label_id in enumerate(predictions):
        if label_id == 1:
            detected_emotions.append(LABELS[i])

    return detected_emotions if detected_emotions else ["neutral"]  # If no emotions are detected, return "neutral"


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Multi-Label Emotion Classifier Prediction")
    parser.add_argument("--text", type=str, required=True, help="The text to analyze")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold for emotion detection")
    args = parser.parse_args()

    print(f"> Text: {args.text}")
    emotions = predict_emotions(args.text, args.threshold)
    print(f"> Emotions: {emotions}")


if __name__ == "__main__":
    main()