import io
import json
import os
import random
import tempfile
import streamlit as st
import torch
from gtts import gTTS
import speech_recognition as sr
from model import clean_and_tokenize, stem_or_lemmatize, bag_of_words, ChatbotNeuralNet

# Page Configuration
st.set_page_config(
    page_title="Voice-Enabled AI Chatbot | SLP Lab Assessment",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .student-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 14px;
        display: inline-block;
        margin-top: 8px;
    }
    .display-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        font-weight: 700;
        color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Metadata
@st.cache_resource
def load_deep_learning_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "model", "chatbot_metadata.json")
    weights_path = os.path.join(base_dir, "model", "chatbot_model.pth")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    loaded_model = ChatbotNeuralNet(
        input_size=meta["input_size"],
        hidden_size_1=meta["hidden_size_1"],
        hidden_size_2=meta["hidden_size_2"],
        hidden_size_3=meta["hidden_size_3"],
        num_classes=meta["num_classes"]
    )
    loaded_model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    loaded_model.eval()
    return loaded_model, meta

model, metadata = load_deep_learning_model()
recognizer = sr.Recognizer()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Voice-Enabled AI Assistant. Record your voice or type your message to begin!"}
    ]
if "last_recognized" not in st.session_state:
    st.session_state.last_recognized = "Awaiting voice input..."
if "last_response" not in st.session_state:
    st.session_state.last_response = "Awaiting query..."
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "None"
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = "0.0%"

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/microphone.png", width=72)
    st.title("🎙️ Voice Chatbot")
    st.markdown("**SLP Lab Assessment (20M)**")
    st.markdown("👨‍🎓 **Hemanth Reddy**  \n🆔 **23BAI0119**")
    st.divider()

    st.subheader("⚙️ Model Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.40, 0.05)
    enable_tts = st.toggle("Enable Voice Reply (TTS)", value=True)

    st.divider()
    st.subheader("📊 System Specs")
    st.markdown("""
    - **Architecture:** 3-Layer Deep Neural Network
    - **Dataset:** 15 Intent Classes (117 Patterns)
    - **Accuracy:** **92.31%** (Macro F1: 0.92)
    - **Speech Engine:** Google Speech Recognition
    - **Framework:** PyTorch & Streamlit
    """)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
        ]
        st.session_state.last_recognized = "Awaiting voice input..."
        st.session_state.last_response = "Awaiting query..."
        st.session_state.last_intent = "None"
        st.session_state.last_confidence = "0.0%"
        st.rerun()

# Main Banner Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:28px;">🎙️ Voice-Enabled Chatbot using Speech Recognition & Deep Learning</h1>
    <p style="margin:6px 0 0 0; font-size:15px; opacity:0.95;">
        Speech and Language Processing (SLP) Lab Assessment Submission | 20 Marks
    </p>
    <div class="student-badge">
        👨‍🎓 <strong>Student:</strong> HEMANTH REDDY &nbsp;|&nbsp; 🆔 <strong>Reg No:</strong> 23BAI0119
    </div>
</div>
""", unsafe_allow_html=True)

# Helper Function: Classify Intent using PyTorch Model
def predict_intent(text: str):
    if not text or not text.strip():
        return "None", 0.0, "No speech detected. Please speak clearly into your microphone."

    tokens = clean_and_tokenize(text)
    bow = bag_of_words(tokens, metadata["vocabulary"])
    bow_tensor = torch.tensor([bow], dtype=torch.float32)

    with torch.no_grad():
        logits = model(bow_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probabilities, dim=0)
    confidence = float(top_prob.item())
    predicted_tag = metadata["classes"][top_idx.item()]

    if confidence < confidence_threshold:
        intent = "fallback"
        responses = metadata["intent_responses"].get("fallback", [
            "I recognized your speech, but my deep learning model has low confidence for that phrase. Please rephrase."
        ])
    else:
        intent = predicted_tag
        responses = metadata["intent_responses"].get(predicted_tag, ["I hear you!"])

    response = random.choice(responses)
    return intent, round(confidence * 100, 2), response

# Helper Function: Generate Audio with gTTS
def generate_audio(text: str):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"TTS generation notice: {e}")
        return None

# Top Section: Mandatory Dual Display (20 Marks Assessment Requirement)
st.subheader("📋 Dual Display: Speech Recognition & Chatbot Response")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎙️ 1. RECOGNIZED SPEECH (Speech-to-Text)")
    st.info(st.session_state.last_recognized)

with col2:
    st.markdown("#### 🤖 2. CHATBOT RESPONSE (Deep Learning Model)")
    st.success(st.session_state.last_response)

# Metrics Bar
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("🧠 Detected Intent", st.session_state.last_intent)
with metric_col2:
    st.metric("📊 DL Confidence", st.session_state.last_confidence)
with metric_col3:
    st.metric("🎯 Model Accuracy", "92.31%")

st.divider()

# Input Section: Voice & Text
st.subheader("🎤 Voice & Text Input")

voice_col, sample_col = st.columns([3, 2])

with voice_col:
    audio_value = st.audio_input("Click the microphone to record your voice:")
    
    if audio_value is not None:
        # Convert audio to wav format for SpeechRecognition
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_value.read())
                tmp_path = tmp_file.name

            with sr.AudioFile(tmp_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source)
                transcribed_text = recognizer.recognize_google(audio_data)

            # Cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            if transcribed_text:
                intent, conf, answer = predict_intent(transcribed_text)
                
                # Update Session State
                st.session_state.last_recognized = transcribed_text
                st.session_state.last_response = answer
                st.session_state.last_intent = intent
                st.session_state.last_confidence = f"{conf}%"

                # Append to Messages
                st.session_state.messages.append({"role": "user", "content": f"🎙️ {transcribed_text}"})
                st.session_state.messages.append({"role": "assistant", "content": f"{answer}  \n*(Intent: `{intent}` | Confidence: `{conf}%`)*"})

                # Optional TTS
                if enable_tts:
                    audio_out = generate_audio(answer)
                    if audio_out:
                        st.audio(audio_out, format="audio/mp3", autoplay=True)

                st.rerun()

        except sr.UnknownValueError:
            st.error("Speech Recognition could not understand the audio. Please speak clearly.")
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")
        except Exception as ex:
            st.warning(f"Audio processing notice: {ex}")

with sample_col:
    st.markdown("**💡 Quick Voice Prompts to Try:**")
    prompt_options = [
        "How does speech recognition work?",
        "Explain your deep learning model architecture",
        "What can you do?",
        "Tell me about this lab assessment",
        "Tell me a funny joke",
        "What is the weather today?"
    ]
    for p in prompt_options:
        if st.button(f"👉 {p}", key=f"btn_{p}", use_container_width=True):
            intent, conf, answer = predict_intent(p)
            st.session_state.last_recognized = p
            st.session_state.last_response = answer
            st.session_state.last_intent = intent
            st.session_state.last_confidence = f"{conf}%"
            st.session_state.messages.append({"role": "user", "content": f"💬 {p}"})
            st.session_state.messages.append({"role": "assistant", "content": f"{answer}  \n*(Intent: `{intent}` | Confidence: `{conf}%`)*"})
            if enable_tts:
                audio_out = generate_audio(answer)
                if audio_out:
                    st.audio(audio_out, format="audio/mp3", autoplay=True)
            st.rerun()

# Fallback Keyboard Input
user_text = st.chat_input("Or type your message here (e.g. 'How does speech recognition work?')...")
if user_text:
    intent, conf, answer = predict_intent(user_text)
    st.session_state.last_recognized = user_text
    st.session_state.last_response = answer
    st.session_state.last_intent = intent
    st.session_state.last_confidence = f"{conf}%"
    st.session_state.messages.append({"role": "user", "content": f"⌨️ {user_text}"})
    st.session_state.messages.append({"role": "assistant", "content": f"{answer}  \n*(Intent: `{intent}` | Confidence: `{conf}%`)*"})
    if enable_tts:
        audio_out = generate_audio(answer)
        if audio_out:
            st.audio(audio_out, format="audio/mp3", autoplay=True)
    st.rerun()

# Multi-Turn Conversation History
st.divider()
st.subheader("💬 Multi-Turn Conversation Transcript")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
