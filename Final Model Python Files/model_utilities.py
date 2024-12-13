from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
import os
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from pathlib import Path

"""
Contains model class and functions for training, predicting, plotting, and calculating metrics
"""

class TransformerBiLSTM(nn.Module):
    def __init__(self, transformer_model="distilbert-base-uncased", hidden_size=128, num_classes=2, dropout=0.5):
        super(TransformerBiLSTM, self).__init__()
        self.transformer = DistilBertModel.from_pretrained(transformer_model)  # Pre-trained DistilBERT
        print("Model loaded successfully!")

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.transformer.config.hidden_size,  # Size of Transformer embeddings
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # BiLSTM is bidirectional

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        # Pass through Transformer
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)
        x = transformer_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Pass through BiLSTM
        lstm_out, _ = self.bilstm(x)  # Shape: (batch_size, seq_len, hidden_size*2)
        lstm_out = lstm_out[:, -1, :]  # Get the last hidden state for classification

        # Fully connected layer
        x = self.dropout(lstm_out)
        logits = self.fc(x)
        return logits


def train_model(
    model, train_loader, test_loader, num_epochs=3, lr=2e-5, device=None, save_model=True, save_dir="models"
):
    """
    Train the model, plot metrics, and optionally save the model.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        num_epochs: Number of epochs.
        lr: Learning rate.
        device: Device for training ('cuda', 'mps', or 'cpu').
        save_model: Whether to save the model after each epoch.
        save_dir: Directory to save the model checkpoints.

    Returns:
        A dictionary containing metrics: train_loss, train_accuracy, test_accuracy.
    """
    # Move model to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cuda' for Nvidia GPUs, 'mps' for Apple Silicon
    model = model.to(device)

    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    # Scheduler for learning rate decay
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Metrics tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # Create directory to save models
    if save_model:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Best test accuracy for saving the best model
    best_test_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss, total_correct = 0, 0
        batch_count = 1

        for batch in train_loader:
            if batch_count % 100 == 0 or batch_count == 1:
                print(f"Processing batch {batch_count}/{len(train_loader)}")

            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

            # Update batch count
            batch_count += 1

        # Calculate average loss and accuracy for training
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # Evaluate on the test set and track test accuracy
        test_accuracy = evaluate_model_with_metrics(model, test_loader, device)
        test_accuracies.append(test_accuracy)

        # Save model after each epoch or when test accuracy improves
        if save_model:
            model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            # Save the best model based on test accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated and saved to {best_model_path}")

    # Plot the results
    plot_training_metrics(train_losses, train_accuracies, test_accuracies, num_epochs)

    return {"train_losses": train_losses, "train_accuracies": train_accuracies, "test_accuracies": test_accuracies}


def evaluate_model_with_metrics(model, data_loader, device):
    """
    Evaluate the model and calculate accuracy on the test set.

    Args:
        model: The trained model.
        data_loader: DataLoader for the test set.
        device: Device for evaluation.

    Returns:
        Test accuracy.
    """
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = total_correct / len(data_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def plot_training_metrics(train_losses, train_accuracies, test_accuracies, num_epochs):
    """
    Plot training loss, training accuracy, and test accuracy per epoch.

    Args:
        train_losses: List of training losses per epoch.
        train_accuracies: List of training accuracies per epoch.
        test_accuracies: List of test accuracies per epoch.
        num_epochs: Number of epochs.
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot training and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(epochs, test_accuracies, label="Test Accuracy", marker="o", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs. Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_with_dropout(model_class, train_loader, test_loader, num_epochs, lr, dropout, device, save_dir="diff_models"):
    """
    Train the model with a specific dropout value and track metrics, saving the model during training.

    Args:
        model_class: The model class to instantiate (e.g., TransformerBiLSTM).
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        num_epochs: Number of epochs to train.
        lr: Learning rate.
        dropout: Dropout rate to apply in the model.
        device: Device for training (e.g., 'cuda', 'mps', or 'cpu').
        save_dir: Directory to save the model checkpoints.

    Returns:
        metrics: Dictionary containing train losses, train accuracies, and test accuracies.
    """

    # Instantiate the model with the specified dropout
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda for windows, mps for apple silicon
    model = model_class(dropout=dropout).to(device)

    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=len(train_loader) * num_epochs)

    # Prepare directory to save models
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_test_accuracy = 0  # Track the best test accuracy for saving the best model

    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}, Dropout: {dropout}")
        model.train()
        total_loss, total_correct = 0, 0

        # track bactch
        batch_count = 1

        for batch in train_loader:
            if batch_count % 100 == 0 or batch_count == 1:
                print(f"Processing batch {batch_count}/{len(train_loader)}")
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            batch_count += 1

        # Calculate average loss and accuracy for training
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on the test set
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        test_accuracy = total_correct / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

        # Save model checkpoint if test accuracy improves
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            model_path = os.path.join(save_dir, f"model_dropout_{dropout}_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
    }


def plot_loss_across_dropouts(metrics_dict):
    """
    Plot training loss across different dropout values.

    Args:
        metrics_dict: Dictionary where keys are dropout values, and values are metrics from training.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Extract training loss and test accuracy for each dropout
    dropouts = []
    losses = []
    accuracies = []
    for dropout, metrics in metrics_dict.items():
        dropouts.append(dropout)
        # Since we're running for 1 epoch only, take the first (and only) training loss and accuracy
        losses.append(metrics["train_losses"][0])
        accuracies.append(metrics["test_accuracies"][0])

    # Plot training loss
    ax1.plot(dropouts, losses, marker='o', color='blue', label="Training Loss")
    ax1.set_xlabel("Dropout Rate")
    ax1.set_ylabel("Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title("Training Loss and Test Accuracy vs. Dropout Rate")

    # Plot test accuracy on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(dropouts, accuracies, marker='o', color='green', label="Test Accuracy")
    ax2.set_ylabel("Accuracy (%)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.grid()
    plt.tight_layout()
    plt.show()

def save_results(all_sequences, all_labels, all_preds, tokenizer, output_dir="results", zip_filename="submission.zip"):
    """
    Save predictions and true labels to a CSV and zip the results.

    Args:
        all_sequences: List of input sequences (token IDs).
        all_labels: List of true labels.
        all_preds: List of predicted labels.
        tokenizer: Tokenizer used to decode token IDs into text.
        output_dir: Directory to save the results.
        zip_filename: Name of the zip file to create.

    Returns:
        csv_path: Path to the saved CSV file.
        zip_path: Path to the saved ZIP file.
    """
    # Decode input sequences back into text using the tokenizer
    decoded_sequences = [
        tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for seq in all_sequences
    ]

    # Create results DataFrame
    results_df = pd.DataFrame({
        'sequence': decoded_sequences,
        'true_label': all_labels,
        'predicted_label': all_preds
    })

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Create a zip file
    zip_path = os.path.join(output_dir, zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(csv_path, arcname="test_predictions.csv")
    print(f"Results zipped to {zip_path}")

    return csv_path, zip_path



def cal_metrics(model, test_loader, tokenizer, output_dir="results", zip_filename="submission.zip"):
    """
    Calculate metrics, save predictions to a CSV, and zip the results.

    Args:
        model: Trained TransformerBiLSTM model.
        test_loader: DataLoader for the test set.
        tokenizer: Tokenizer used for decoding sequences.
        output_dir: Directory to save the results.
        zip_filename: Name of the zip file to create.

    Returns:
        f1: F1-score of the model.
        precision: Precision of the model.
        recall: Recall of the model.
        accuracy: Accuracy of the model.
        zip_path: Path to the saved ZIP file.
    """
    device = next(model.parameters()).device  # Get device (e.g., 'cuda', 'mps', 'cpu')
    model.eval()  # Set model to evaluation mode

    # Initialize lists for predictions, labels, and sequences
    all_preds = []
    all_labels = []
    all_sequences = []

    # Collect predictions and true labels
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sequences.extend(input_ids.cpu().numpy())  # Save raw input sequences

    # Calculate metrics
    tp = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label == 1)
    fp = sum(1 for pred, label in zip(all_preds, all_labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(all_preds, all_labels) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label == 0)

    accuracy = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Test Set Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}")

    # Save results using the helper function
    csv_path, zip_path = save_results(all_sequences, all_labels, all_preds, tokenizer, output_dir, zip_filename)

    return f1, precision, recall, accuracy, zip_path


def read_csv_for_out_of_sample(csv_path, text_column="text", label_column="label"):
    """
    Reads a CSV file and extracts text and labels for out-of-sample preprocessing.

    Args:
        csv_path (str): Path to the CSV file.
        text_column (str): Column name containing the text data.
        label_column (str): Column name containing the labels.

    Returns:
        data (list[str]): List of text samples.
        labels (list[int]): List of corresponding sentiment labels.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Extract text and labels
    data = df[text_column].tolist()
    labels = df[label_column].tolist()

    return data, labels


def preprocess_out_of_sample(data, labels, batch_size=32, num_steps=500, device=None):
    """
    Prepares an out-of-sample dataset for the Transformer + BiLSTM model.
    Tokenizes, truncates/pads sequences, and creates a DataLoader.

    Args:
        data (list[str]): List of text reviews in the out-of-sample set.
        labels (list[int]): Corresponding sentiment labels (e.g., 0 for negative, 1 for positive).
        batch_size (int): Batch size for the DataLoader (default: 32).
        num_steps (int): Maximum length for truncation/padding (default: 500).
        device (torch.device): Device to move tensors to ('cpu', 'cuda', or 'mps').

    Returns:
        data_loader (DataLoader): DataLoader for the out-of-sample set.
        tokenizer (DistilBertTokenizer): Tokenizer used for preprocessing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenize and encode sequences
    encoded_data = tokenizer(
        data,
        truncation=True,
        padding="max_length",
        max_length=num_steps,
        return_tensors="pt"
    )

    # Move tokenized data to the specified device
    encoded_data = {key: val.to(device) for key, val in encoded_data.items()}

    # Convert labels to tensors and move to the device
    labels_tensor = torch.tensor(labels).to(device)

    # Create a TensorDataset
    dataset = TensorDataset(encoded_data["input_ids"], encoded_data["attention_mask"], labels_tensor)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader, tokenizer


def predict_sentiment(model, tokenizer, text, device=None, max_length=500):
    """
    Predict the sentiment of a given text using the trained model.

    Args:
        model: Trained TransformerBiLSTM model.
        tokenizer: Pretrained tokenizer (e.g., DistilBERT tokenizer).
        text: The input text string to classify.
        device: Device for computation ('cuda', 'mps', or 'cpu').
        max_length: Maximum length for tokenization (default: 500).

    Returns:
        Predicted sentiment label (e.g., 'positive' or 'negative').
    """
    # Move model to the correct device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cuda' for Nvidia GPUs, 'mps' for Apple Silicon
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Tokenize and encode the input text
    encoded_input = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Move input tensors to the same device as the model
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_label = outputs.argmax(dim=1).item()  # Get the index of the highest score

    # Map predicted label to sentiment class (e.g., 0 -> 'negative', 1 -> 'positive')
    sentiment = "positive" if predicted_label == 1 else "negative"
    return sentiment