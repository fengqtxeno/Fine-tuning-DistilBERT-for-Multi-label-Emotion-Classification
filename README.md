# Emotion-Aware NLP: Fine-tuning DistilBERT for Multi-label Emotion Classification

This project demonstrates a complete NLP workflow from data processing and model fine-tuning to inference applications. We focus on a more advanced task than simple sentiment analysis: multi-label emotion classification.

## Project Highlights

- **Model Fine-tuning**: Leverages Hugging Face transformers and the Trainer API to fine-tune a `distilbert-base-uncased` model
- **Multi-label Classification**: The model can identify multiple emotions present in text (not just a single one)
- **Emotion-Aware Dataset**: Uses the GoEmotions dataset, which contains 27 emotion categories plus 1 neutral category, perfectly aligned with emotionally aware NLP research
- **End-to-End Workflow**:
  - `src/data_loader.py`: Automatically downloads and preprocesses data
  - `src/train.py`: Runs model fine-tuning and evaluation
  - `src/predict.py`: Quick inference script for command-line usage
  - `src/gui.py`: A simple PyQt5 GUI application for interacting with the fine-tuned model

## Project Structure

```
.
├── models/                     # Trained models will be saved here
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── train.py                # Model fine-tuning script
│   ├── predict.py              # Command-line inference script
│   └── gui.py                  # GUI application
├── README.md                   # This document
└── requirements.txt            # Python dependencies
```

## Getting Started

### 1. Environment Setup

It's recommended to use a virtual environment:

```bash
  python -m venv venv
  source venv/bin/activate  # on Windows: venv\Scripts\activate
```

Install all dependencies:

```bash
  pip install -r requirements.txt
```

### 2. Train the Model

Run the training script. This will automatically download the GoEmotions dataset, preprocess it, and begin fine-tuning. The model and tokenizer will be saved in the `./models/emotion_classifier/` directory.

```bash
  python src/train.py
```

**Note**: Training requires a CUDA-enabled GPU to complete in a reasonable time.

### 3. Command-line Prediction

Once training is complete, you can test the model using `predict.py`:

```bash
  python src/predict.py --text "I am so happy and excited for this new journey!"
```

**Expected output**:
```
> Text: I am so happy and excited for this new journey!
> Emotions: ['admiration', 'excitement', 'joy', 'optimism']
```

### 4. Run the GUI Application

You can also interact with the model through a graphical interface:

```bash
  python src/gui.py
```

## Dataset

This project uses the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset, a corpus of 58k carefully curated Reddit comments labeled for 27 emotion categories. The dataset is particularly well-suited for training emotionally aware NLP models.

## Model Architecture

The project uses DistilBERT, a distilled version of BERT that retains 97% of BERT's language understanding while being 60% faster and 40% smaller. This makes it an excellent choice for practical applications while maintaining strong performance on emotion classification tasks.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- PyQt5 (for GUI)
- scikit-learn
- numpy

See `requirements.txt` for complete dependency list.


