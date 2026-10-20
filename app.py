import os
import json
import random
import gradio as gr
import speech_recognition as sr
import torch
from model import clean_and_tokenize, stem_or_lemmatize, bag_of_words, ChatbotNeuralNet

# Load model and metadata
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
METADATA_PATH = os.path.join(MODEL_DIR, "chatbot_metadata.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "chatbot_model.pth")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

vocabulary = metadata["vocabulary"]
classes = metadata["classes"]
intent_responses = metadata["intent_responses"]
confidence_threshold = metadata.get("confidence_threshold", 0.40)

# Initialize PyTorch Deep Learning Model
model = ChatbotNeuralNet(
    input_size=metadata["input_size"],
    hidden_size_1=metadata["hidden_size_1"],
    hidden_size_2=metadata["hidden_size_2"],
    hidden_size_3=metadata["hidden_size_3"],
    num_classes=metadata["num_classes"]
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu")))
model.eval()

recognizer = sr.Recognizer()

def classify_intent_and_respond(user_text: str):
    """Classifies user utterance using the trained Deep Neural Network."""
    if not user_text or not user_text.strip():
        return "None detected", "Please speak into your microphone or type a message.", "0.0%", "N/A"

    tokens = clean_and_tokenize(user_text)
    bow = bag_of_words(tokens, vocabulary)
    bow_tensor = torch.tensor([bow], dtype=torch.float32)

    with torch.no_grad():
        logits = model(bow_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probabilities, dim=0)
    confidence = float(top_prob.item())
    predicted_tag = classes[top_idx.item()]

    if confidence < confidence_threshold:
        responses = intent_responses.get("fallback", [
            "I recognized your speech, but the intent was below the confidence threshold. Please rephrase."
        ])
        intent_display = "fallback (Out of Domain)"
    else:
        responses = intent_responses.get(predicted_tag, ["I hear you!"])
        intent_display = predicted_tag

    selected_response = random.choice(responses)
    confidence_display = f"{confidence * 100:.2f}%"

    return user_text, selected_response, confidence_display, intent_display

def process_voice_input(audio_file_path, chat_history):
    """Processes speech audio file using Speech Recognition and the Deep Learning model."""
    if chat_history is None:
        chat_history = []

    if audio_file_path is None:
        return (
            "No audio input received. Click the microphone button and record your voice.",
            "Awaiting your voice input...",
            "0.0%",
            "None",
            chat_history
        )

    recognized_text = ""
    try:
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        recognized_text = "[Speech Recognition could not transcribe the audio. Please speak more clearly.]"
        rec_speech, response, conf, intent = (
            recognized_text,
            "I heard sound, but could not decode the phonemes clearly into words. Please try again with clear pronunciation.",
            "0.0%",
            "speech_recognition_error"
        )
        chat_history.append({"role": "user", "content": "🎙️ [Unintelligible Voice Input]"})
        chat_history.append({"role": "assistant", "content": response})
        return rec_speech, response, conf, intent, chat_history
    except sr.RequestError as e:
        rec_speech, response, conf, intent = (
            f"[Speech Recognition API error: {e}]",
            "Speech recognition service is currently unreachable. Please use text input or check connection.",
            "0.0%",
            "api_error"
        )
        return rec_speech, response, conf, intent, chat_history

    rec_speech, response, conf, intent = classify_intent_and_respond(recognized_text)
    chat_history.append({"role": "user", "content": f"🎙️ {rec_speech}"})
    chat_history.append({"role": "assistant", "content": f"🤖 {response} (Intent: {intent} | Conf: {conf})"})

    return rec_speech, response, conf, intent, chat_history

def process_text_input(text_message, chat_history):
    """Processes typed text query for hybrid accessibility."""
    if chat_history is None:
        chat_history = []
    if not text_message or not text_message.strip():
        return "", "", "0.0%", "None", chat_history

    rec_speech, response, conf, intent = classify_intent_and_respond(text_message)
    chat_history.append({"role": "user", "content": f"⌨️ {rec_speech}"})
    chat_history.append({"role": "assistant", "content": f"🤖 {response} (Intent: {intent} | Conf: {conf})"})

    return rec_speech, response, conf, intent, chat_history

# Gradio Interface
custom_css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
}
"""

with gr.Blocks(title="Voice-Enabled Deep Learning Chatbot | SLP Lab Assessment") as demo:
    gr.HTML("""
    <div class="header-box">
        <h1 style="margin:0; font-size:26px;">🎙️ Voice-Enabled Chatbot using Speech Recognition & Deep Learning</h1>
        <p style="margin:6px 0 0 0; font-size:15px; opacity:0.95; font-weight:500;">
            Speech and Language Processing (SLP) Lab Assessment Submission | 20 Marks
        </p>
        <p style="margin:4px 0 0 0; font-size:14px; opacity:0.9; color:#fef08a;">
            <strong>Student:</strong> HEMANTH REDDY &nbsp;|&nbsp; <strong>Reg No:</strong> 23BAI0119
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1️⃣ Voice & Text Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speak into your Microphone (Click record, speak, then stop)"
            )
            voice_btn = gr.Button("🎤 Process Voice Input", variant="primary", size="lg")

            gr.Markdown("---")
            text_input = gr.Textbox(
                placeholder="Or type your query here (e.g. 'How does speech recognition work?')...",
                label="Fallback / Hybrid Text Input"
            )
            text_btn = gr.Button("💬 Send Text Query", variant="secondary")

            gr.Markdown("#### 💡 Quick Voice Prompts to Try:")
            gr.Examples(
                examples=[
                    ["Hello there, good morning!"],
                    ["How does speech recognition work?"],
                    ["Explain your deep learning model architecture"],
                    ["What are your features and capabilities?"],
                    ["What is this 20 marks lab assessment about?"],
                    ["Tell me a funny joke"],
                    ["What is the weather today?"],
                    ["Thank you very much!"]
                ],
                inputs=[text_input]
            )

        with gr.Column(scale=6):
            gr.Markdown("### 2️⃣ Speech & Deep Learning Response Output")

            recognized_box = gr.Textbox(
                label="🎙️ 1. RECOGNIZED SPEECH (Speech-to-Text Transcription)",
                placeholder="Spoken words will appear here...",
                lines=2,
                interactive=False
            )

            response_box = gr.Textbox(
                label="🤖 2. CHATBOT RESPONSE (Generated Answer)",
                placeholder="Chatbot response will appear here...",
                lines=3,
                interactive=False
            )

            with gr.Row():
                intent_box = gr.Textbox(label="🧠 Detected Intent Tag", interactive=False)
                conf_box = gr.Textbox(label="📊 Deep Learning Confidence", interactive=False)

            chatbot_history = gr.Chatbot(label="💬 Full Conversation Transcript", height=320)

    # Event handlers
    voice_btn.click(
        fn=process_voice_input,
        inputs=[audio_input, chatbot_history],
        outputs=[recognized_box, response_box, conf_box, intent_box, chatbot_history]
    )

    text_btn.click(
        fn=process_text_input,
        inputs=[text_input, chatbot_history],
        outputs=[recognized_box, response_box, conf_box, intent_box, chatbot_history]
    )

    text_input.submit(
        fn=process_text_input,
        inputs=[text_input, chatbot_history],
        outputs=[recognized_box, response_box, conf_box, intent_box, chatbot_history]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
