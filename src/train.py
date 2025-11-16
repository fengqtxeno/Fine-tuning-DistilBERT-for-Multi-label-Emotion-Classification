import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from src.data_loader import load_and_preprocess_data, MODEL_NAME, NUM_LABELS, id2label, label2id

# Define model and training parameters
OUTPUT_DIR = "./models/emotion_classifier"
BATCH_SIZE = 16
EPOCHS = 3  # 3-5 epochs recommended for full training, set to 3 for quick demo
LEARNING_RATE = 2e-5


def compute_metrics(eval_pred):
    """
    Custom compute_metrics function for multi-label classification.
    """
    logits, labels = eval_pred

    # Key: Use Sigmoid for multi-label classification, not Softmax
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.from_numpy(logits))

    # Use a threshold of 0.5 to determine predicted labels
    predictions = (probs > 0.5).int().numpy()

    # Calculate metrics
    # 'micro' F1-score is suitable for multi-label imbalanced datasets
    f1 = f1_score(labels, predictions, average='micro')
    # 'weighted' ROC AUC
    try:
        roc_auc = roc_auc_score(labels, probs, average='weighted')
    except ValueError:
        roc_auc = 0.0  # In case some batches lack sufficient labels

    # 'subset accuracy' (requires all labels to be perfectly correct)
    subset_accuracy = accuracy_score(labels, predictions)

    return {
        'f1_micro': f1,
        'roc_auc_weighted': roc_auc,
        'subset_accuracy': subset_accuracy
    }


def main():
    print(f"Using model: {MODEL_NAME}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load and preprocess data
    train_dataset, eval_dataset, labels_list = load_and_preprocess_data(tokenizer)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(eval_dataset)}")

    # 3. Load Model
    print("Loading pretrained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        # Key: Inform the model this is a multi-label classification problem
        # This will make the model use Sigmoid activation and BCEWithLogitsLoss
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )

    # 4. Define Training Arguments
    # TrainingArguments contains all training configurations
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,  # Log to console every 500 steps
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch
        load_best_model_at_end=True,  # Load the best model found on the validation set at the end of training
        metric_for_best_model="f1_micro",  # Use f1_micro as the metric for the best model
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # If a GPU is available, use fp16 for faster training
        report_to="none"  # Disable reporting to wandb/tensorboard etc.
    )

    # 5. Initialize Trainer
    # The Trainer API greatly simplifies the PyTorch training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Pass in our custom compute_metrics function
    )

    # 6. Start Training
    print("\n--- Starting Model Fine-Tuning ---")
    trainer.train()
    print("--- Model Fine-Tuning Complete ---")

    # 7. Evaluate Model
    print("\n--- Evaluating Final Model on Validation Set ---")
    eval_results = trainer.evaluate()
    print(eval_results)

    # 8. Save Final Model and Tokenizer
    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saving complete.")


if __name__ == "__main__":
    main()