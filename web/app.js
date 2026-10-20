/**
 * Voice-Enabled Chatbot Frontend Application
 * Integrates Web Speech Recognition API, Deep Learning Inference Engine,
 * and Speech Synthesis for complete voice-in, voice-out interaction.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const micBtn = document.getElementById("micBtn");
    const micIcon = document.getElementById("micIcon");
    const statusBadge = document.getElementById("statusBadge");
    const micInstructions = document.getElementById("micInstructions");
    const audioWave = document.getElementById("audioWave");
    const textInput = document.getElementById("textInput");
    const sendBtn = document.getElementById("sendBtn");
    const speechOutputText = document.getElementById("speechOutputText");
    const responseOutputText = document.getElementById("responseOutputText");
    const detectedIntent = document.getElementById("detectedIntent");
    const confidenceScore = document.getElementById("confidenceScore");
    const speechLatency = document.getElementById("speechLatency");
    const chatHistory = document.getElementById("chatHistory");
    const ttsToggle = document.getElementById("ttsToggle");
    const copySpeechBtn = document.getElementById("copySpeechBtn");
    const speakResponseBtn = document.getElementById("speakResponseBtn");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const promptChips = document.querySelectorAll(".chip");

    let isRecording = false;
    let recognition = null;
    let speechStartTime = 0;
    let latestResponseText = "";

    // Initialize Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onstart = () => {
            isRecording = true;
            speechStartTime = performance.now();
            micBtn.classList.add("listening");
            audioWave.classList.remove("inactive");
            statusBadge.className = "status-badge listening";
            statusBadge.innerHTML = '<i class="fa-solid fa-microphone-lines"></i> Listening to Voice...';
            micInstructions.textContent = "Listening... Speak your query clearly now.";
            speechOutputText.classList.remove("placeholder");
            speechOutputText.textContent = "Transcribing speech in real-time...";
        };

        recognition.onresult = (event) => {
            let interimTranscript = "";
            let finalTranscript = "";

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            if (finalTranscript) {
                const latency = Math.round(performance.now() - speechStartTime);
                speechLatency.textContent = `${latency} ms`;
                handleUserQuery(finalTranscript, true);
            } else if (interimTranscript) {
                speechOutputText.textContent = interimTranscript;
            }
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            isRecording = false;
            resetMicUI();

            if (event.error === "not-allowed" || event.error === "permission-denied") {
                alert("Microphone permission denied! Please allow microphone access in your browser settings to use voice input, or use the text box below.");
                statusBadge.className = "status-badge";
                statusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Mic Permission Denied';
            } else if (event.error === "no-speech") {
                statusBadge.className = "status-badge ready";
                statusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> Ready to Listen';
                micInstructions.textContent = "No voice heard. Please click the mic and try speaking again.";
            } else {
                statusBadge.className = "status-badge";
                statusBadge.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> Error: ${event.error}`;
            }
        };

        recognition.onend = () => {
            isRecording = false;
            resetMicUI();
        };
    } else {
        console.warn("Web Speech API not supported in this browser. Falling back to text input.");
        micInstructions.textContent = "Voice input is not supported in this browser. Please use Chrome/Edge or type below.";
        statusBadge.className = "status-badge";
        statusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Speech API Not Supported';
    }

    function resetMicUI() {
        micBtn.classList.remove("listening");
        audioWave.classList.add("inactive");
        statusBadge.className = "status-badge ready";
        statusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> Ready to Listen';
        micInstructions.textContent = "Click the microphone button and speak your query";
    }

    // Toggle Microphone Recording
    micBtn.addEventListener("click", () => {
        if (!recognition) {
            alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or an Android browser.");
            return;
        }

        if (isRecording) {
            recognition.stop();
        } else {
            try {
                recognition.start();
            } catch (err) {
                console.error("Failed to start speech recognition:", err);
            }
        }
    });

    // Handle User Query (from voice or text)
    function handleUserQuery(queryText, fromVoice = false) {
        if (!queryText || !queryText.trim()) return;

        const startTime = performance.now();

        // 1. Update Recognized Speech Display
        speechOutputText.classList.remove("placeholder");
        speechOutputText.textContent = queryText.trim();

        // 2. Deep Learning Neural Network Inference
        const result = predictDeepLearningIntent(queryText.trim());
        const totalLatency = Math.round(performance.now() - startTime);

        if (!fromVoice) {
            speechLatency.textContent = `${totalLatency} ms (keyboard)`;
        }

        // 3. Update Chatbot Response Display
        responseOutputText.classList.remove("placeholder");
        responseOutputText.textContent = result.response;
        detectedIntent.textContent = result.intent;
        confidenceScore.textContent = `${result.confidence}%`;

        latestResponseText = result.response;

        // 4. Append to Conversation History
        appendMessageToHistory(queryText, "user", fromVoice);
        appendMessageToHistory(result.response, "assistant", false, result.intent, result.confidence);

        // 5. Text-To-Speech Synthesis (Voice Response)
        if (ttsToggle.checked && "speechSynthesis" in window) {
            speakText(result.response);
        }
    }

    // Append Message to History Transcript
    function appendMessageToHistory(text, sender, isVoice = false, intent = null, conf = null) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message`;

        const avatarDiv = document.createElement("div");
        avatarDiv.className = "msg-avatar";
        avatarDiv.innerHTML = sender === "user"
            ? (isVoice ? '<i class="fa-solid fa-microphone"></i>' : '<i class="fa-solid fa-user"></i>')
            : '<i class="fa-solid fa-robot"></i>';

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "msg-bubble";

        let intentTag = "";
        if (sender === "assistant" && intent) {
            intentTag = ` <span style="font-size:11px; opacity:0.75; font-family:'JetBrains Mono',monospace;">[${intent} | ${conf}%]</span>`;
        }

        bubbleDiv.innerHTML = `
            <p>${text}${intentTag}</p>
            <span class="msg-time">${timeStr}</span>
        `;

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(bubbleDiv);

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Text to Speech
    function speakText(text) {
        window.speechSynthesis.cancel(); // Cancel any ongoing speech
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.lang = "en-US";
        window.speechSynthesis.speak(utterance);
    }

    // Text Input Events
    sendBtn.addEventListener("click", () => {
        const text = textInput.value;
        if (text.trim()) {
            handleUserQuery(text, false);
            textInput.value = "";
        }
    });

    textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const text = textInput.value;
            if (text.trim()) {
                handleUserQuery(text, false);
                textInput.value = "";
            }
        }
    });

    // Suggested Prompt Chips
    promptChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            textInput.value = query;
            handleUserQuery(query, false);
        });
    });

    // Copy Speech Button
    copySpeechBtn.addEventListener("click", () => {
        const text = speechOutputText.textContent;
        if (text && !speechOutputText.classList.contains("placeholder")) {
            navigator.clipboard.writeText(text).then(() => {
                copySpeechBtn.innerHTML = '<i class="fa-solid fa-check" style="color:#10b981;"></i>';
                setTimeout(() => {
                    copySpeechBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
                }, 1500);
            });
        }
    });

    // Replay Response Audio Button
    speakResponseBtn.addEventListener("click", () => {
        if (latestResponseText) {
            speakText(latestResponseText);
        }
    });

    // Clear Chat
    clearChatBtn.addEventListener("click", () => {
        chatHistory.innerHTML = `
            <div class="message assistant-message">
                <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="msg-bubble">
                    <p>Chat history cleared. Click the microphone button and ask me anything!</p>
                    <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
            </div>
        `;
        speechOutputText.classList.add("placeholder");
        speechOutputText.textContent = "Awaiting your voice input... (Speak into the microphone)";
        responseOutputText.classList.add("placeholder");
        responseOutputText.textContent = "The chatbot's generated answer will appear here...";
        detectedIntent.textContent = "None";
        confidenceScore.textContent = "0.0%";
        speechLatency.textContent = "0 ms";
    });
});
