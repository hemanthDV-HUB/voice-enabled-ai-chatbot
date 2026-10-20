import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from model import clean_and_tokenize, stem_or_lemmatize, bag_of_words, ChatbotNeuralNet

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class IntentDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)
        self.n_samples = len(X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def train_chatbot_model():
    set_seed(42)
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "intents.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    vocabulary = []
    classes = []
    documents = [] # list of (tokenized_words, tag)
    intent_responses = {}

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        classes.append(tag)
        intent_responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            tokens = clean_and_tokenize(pattern)
            stems = [stem_or_lemmatize(t) for t in tokens]
            vocabulary.extend(stems)
            documents.append((tokens, tag))

    # Deduplicate and sort vocabulary and classes
    vocabulary = sorted(list(set(vocabulary)))
    classes = sorted(list(set(classes)))

    print(f"Dataset Loaded Successfully!")
    print(f"Total Patterns/Documents: {len(documents)}")
    print(f"Distinct Intent Classes ({len(classes)}): {classes}")
    print(f"Vocabulary Size (|V|): {len(vocabulary)}")

    # Create training vectors
    X = []
    y = []
    for (tokens, tag) in documents:
        bow = bag_of_words(tokens, vocabulary)
        X.append(bow)
        label = classes.index(tag)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Train / Test split (stratified shuffle)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_idx = int(0.85 * len(X))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # DataLoader
    batch_size = 16
    train_dataset = IntentDataset(X_train, y_train)
    val_dataset = IntentDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_size = len(vocabulary)
    num_classes = len(classes)
    hidden_1 = 128
    hidden_2 = 64
    hidden_3 = 32

    model = ChatbotNeuralNet(
        input_size=input_size,
        hidden_size_1=hidden_1,
        hidden_size_2=hidden_2,
        hidden_size_3=hidden_3,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    num_epochs = 260
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print("\n--- Starting Deep Learning Model Training ---")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        scheduler.step()

        train_loss = total_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        val_acc = correct_val / total_val if total_val > 0 else 1.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 30 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    # Final Evaluation across full dataset
    model.eval()
    with torch.no_grad():
        full_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(full_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probabilities, axis=1)

    overall_acc = accuracy_score(y, preds)
    print("\n=== Final Deep Learning Evaluation Report ===")
    print(f"Overall Dataset Accuracy: {overall_acc * 100:.2f}%")
    report = classification_report(y, preds, target_names=classes, zero_division=0)
    print(report)

    # Save model and artifacts
    os.makedirs("model", exist_ok=True)
    model_save_path = os.path.join("model", "chatbot_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model state dict saved to: {model_save_path}")

    # Export metadata
    metadata = {
        "vocabulary": vocabulary,
        "classes": classes,
        "intent_responses": intent_responses,
        "input_size": input_size,
        "hidden_size_1": hidden_1,
        "hidden_size_2": hidden_2,
        "hidden_size_3": hidden_3,
        "num_classes": num_classes,
        "overall_accuracy": round(float(overall_acc), 4),
        "confidence_threshold": 0.40
    }
    metadata_save_path = os.path.join("model", "chatbot_metadata.json")
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_save_path}")

    # Export weights to JSON for high-portability browser & serverless execution
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
    for name, buffer in model.named_buffers():
        weights_dict[name] = buffer.detach().cpu().numpy().tolist()

    weights_save_path = os.path.join("model", "model_weights.json")
    with open(weights_save_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f)
    print(f"Neural Network weights exported to: {weights_save_path}")

    return metadata

if __name__ == "__main__":
    train_chatbot_model()
