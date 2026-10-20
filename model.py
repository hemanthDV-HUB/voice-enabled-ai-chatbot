import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F

def clean_and_tokenize(text: str) -> list[str]:
    """Tokenize and normalize text by lowercasing and removing punctuation."""
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Tokenize words
    tokens = text.strip().split()
    return tokens

def stem_or_lemmatize(word: str) -> str:
    """Lightweight rule-based stemmer for natural language tokens."""
    word = word.lower()
    # Simple suffix stripping for common English endings
    suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ("ing", ""), ("ies", "y"), ("es", ""), ("s", ""), ("ed", "")
    ]
    for suffix, replacement in suffixes:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)] + replacement
    return word

def bag_of_words(tokenized_sentence: list[str], vocabulary: list[str]) -> list[float]:
    """Create a bag-of-words binary/count feature vector for the given vocabulary."""
    sentence_stems = [stem_or_lemmatize(w) for w in tokenized_sentence]
    bag = [0.0] * len(vocabulary)
    for idx, vocab_word in enumerate(vocabulary):
        if vocab_word in sentence_stems:
            bag[idx] = 1.0
    return bag

class ChatbotNeuralNet(nn.Module):
    """Deep Neural Network for Intent Classification in Voice-Enabled Chatbot."""
    def __init__(self, input_size: int, hidden_size_1: int = 128, hidden_size_2: int = 64, hidden_size_3: int = 32, num_classes: int = 15):
        super(ChatbotNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(0.35)
        
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout2 = nn.Dropout(0.30)
        
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.dropout3 = nn.Dropout(0.20)
        
        self.out = nn.Linear(hidden_size_3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.fc1(x)
        if (not self.training) or (x.size(0) > 1):
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if (not self.training) or (x.size(0) > 1):
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer (logits)
        out = self.out(x)
        return out
