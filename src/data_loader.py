import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Define the pretrained model we will use
MODEL_NAME = "distilbert-base-uncased"
# GoEmotions dataset has 28 labels
NUM_LABELS = 28
# List of labels (from GoEmotions official documentation)
LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
# Create label -> id and id -> label mappings
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}


def load_and_preprocess_data(tokenizer):
    """
    Loads and preprocesses the GoEmotions dataset.
    """
    print("Loading GoEmotions dataset (simplified)...")
    # Load the "simplified" version of the dataset
    dataset = load_dataset("go_emotions", "simplified")

    print("Dataset info:")
    print(dataset)

    def preprocess_function(examples):
        """
        Tokenize text and convert labels.
        """
        # Tokenize the texts
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

        # Convert label lists to multi-hot encoded vectors
        # This is a crucial step as the original dataset's labels are lists like [3, 15]
        labels = torch.zeros(len(examples["text"]), NUM_LABELS)
        for i, label_indices in enumerate(examples["labels"]):
            for idx in label_indices:
                labels[i, idx] = 1.0

        tokenized_inputs["labels"] = labels.tolist()
        return tokenized_inputs

    print("Tokenizing and preprocessing dataset...")
    # 'batched=True' speeds up processing
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Remove original 'text' and 'labels' columns as they are no longer needed
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "labels", "id"])
    # Set dataset format to PyTorch tensors
    tokenized_datasets.set_format("torch")

    print("Preprocessing complete.")
    return tokenized_datasets["train"], tokenized_datasets["validation"], LABELS


if __name__ == "__main__":
    # As a simple test, load tokenizer and run the data loader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, labels = load_and_preprocess_data(tokenizer)
    print(f"\nTraining set samples: {len(train_ds)}")
    print(f"Validation set samples: {len(val_ds)}")
    print(f"\nFirst training sample: \n{train_ds[0]}")
    print(f"\nLabel list: \n{labels}")