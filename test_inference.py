import json
import os
import random
import torch
import numpy as np
from model import clean_and_tokenize, stem_or_lemmatize, bag_of_words, ChatbotNeuralNet

class VoiceChatbotEngine:
    def __init__(self, model_dir="model"):
        metadata_path = os.path.join(model_dir, "chatbot_metadata.json")
        model_path = os.path.join(model_dir, "chatbot_model.pth")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.vocabulary = self.metadata["vocabulary"]
        self.classes = self.metadata["classes"]
        self.intent_responses = self.metadata["intent_responses"]
        self.confidence_threshold = self.metadata.get("confidence_threshold", 0.40)

        self.model = ChatbotNeuralNet(
            input_size=self.metadata["input_size"],
            hidden_size_1=self.metadata["hidden_size_1"],
            hidden_size_2=self.metadata["hidden_size_2"],
            hidden_size_3=self.metadata["hidden_size_3"],
            num_classes=self.metadata["num_classes"]
        )
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def predict(self, text: str):
        if not text or not text.strip():
            return {
                "recognized_text": "",
                "intent": "empty",
                "confidence": 0.0,
                "response": "I didn't hear or receive any input. Please speak or type your question."
            }

        tokens = clean_and_tokenize(text)
        bow = bag_of_words(tokens, self.vocabulary)
        bow_tensor = torch.tensor([bow], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(bow_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        top_prob, top_idx = torch.max(probabilities, dim=0)
        confidence = float(top_prob.item())
        tag = self.classes[top_idx.item()]

        if confidence < self.confidence_threshold:
            fallback_responses = self.intent_responses.get("fallback", [
                "I recognized your speech, but could not determine the exact intent with high confidence. Please rephrase."
            ])
            response = random.choice(fallback_responses)
            return {
                "recognized_text": text,
                "intent": "low_confidence_fallback",
                "confidence": round(confidence * 100, 2),
                "response": response
            }

        responses = self.intent_responses.get(tag, ["I hear you!"])
        chosen_response = random.choice(responses)
        return {
            "recognized_text": text,
            "intent": tag,
            "confidence": round(confidence * 100, 2),
            "response": chosen_response
        }

if __name__ == "__main__":
    engine = VoiceChatbotEngine()
    test_queries = [
        "Hello there, good morning!",
        "How does speech recognition convert my voice to text?",
        "Explain your deep learning model architecture",
        "What are your features and capabilities?",
        "What is this 20 marks lab assessment about?",
        "Tell me a funny joke",
        "What is the weather outside?",
        "Thank you very much!",
        "Goodbye, see you tomorrow",
        "xyzzzqwer banana astronaut"
    ]

    print("\n--- Testing Voice Chatbot Deep Learning Inference Engine ---")
    for q in test_queries:
        res = engine.predict(q)
        print(f"\nUser Input: '{q}'")
        print(f"-> Detected Intent: {res['intent']} (Confidence: {res['confidence']}%)")
        print(f"-> Chatbot Response: {res['response']}")
