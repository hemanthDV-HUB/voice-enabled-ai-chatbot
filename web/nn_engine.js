/**
 * Deep Learning Neural Network Inference Engine (Pure JS)
 * Exact mathematical forward pass matching PyTorch ChatbotNeuralNet
 */

function cleanAndTokenize(text) {
    if (!text) return [];
    text = text.toLowerCase();
    // Remove punctuation
    text = text.replace(/[\.,-\/#!$%\^&\*;:{}=\-_`~()?"'’]/g, " ");
    return text.trim().split(/\s+/).filter(w => w.length > 0);
}

function stemOrLemmatize(word) {
    word = word.toLowerCase();
    const suffixes = [
        ["ational", "ate"], ["tional", "tion"], ["enci", "ence"], ["anci", "ance"],
        ["izer", "ize"], ["bli", "ble"], ["alli", "al"], ["entli", "ent"],
        ["eli", "e"], ["ousli", "ous"], ["ization", "ize"], ["ation", "ate"],
        ["ator", "ate"], ["alism", "al"], ["iveness", "ive"], ["fulness", "ful"],
        ["ousness", "ous"], ["aliti", "al"], ["iviti", "ive"], ["biliti", "ble"],
        ["ing", ""], ["ies", "y"], ["es", ""], ["s", ""], ["ed", ""]
    ];
    for (const [suffix, replacement] of suffixes) {
        if (word.length > suffix.length + 2 && word.endsWith(suffix)) {
            return word.slice(0, -suffix.length) + replacement;
        }
    }
    return word;
}

function bagOfWords(tokens, vocabulary) {
    const stems = tokens.map(t => stemOrLemmatize(t));
    const bag = new Float32Array(vocabulary.length);
    for (let i = 0; i < vocabulary.length; i++) {
        if (stems.includes(vocabulary[i])) {
            bag[i] = 1.0;
        }
    }
    return bag;
}

function linear(x, weights, bias) {
    const outDim = bias.length;
    const inDim = x.length;
    const out = new Float32Array(outDim);
    for (let i = 0; i < outDim; i++) {
        let sum = bias[i];
        const row = weights[i];
        for (let j = 0; j < inDim; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
    return out;
}

function batchnorm(x, mean, variance, weight, bias, eps = 1e-5) {
    const len = x.length;
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        const std = Math.sqrt(variance[i] + eps);
        out[i] = ((x[i] - mean[i]) / std) * weight[i] + bias[i];
    }
    return out;
}

function relu(x) {
    const len = x.length;
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        out[i] = x[i] > 0 ? x[i] : 0;
    }
    return out;
}

function softmax(x) {
    const len = x.length;
    let max = -Infinity;
    for (let i = 0; i < len; i++) {
        if (x[i] > max) max = x[i];
    }
    const exps = new Float32Array(len);
    let sum = 0;
    for (let i = 0; i < len; i++) {
        const exp = Math.exp(x[i] - max);
        exps[i] = exp;
        sum += exp;
    }
    for (let i = 0; i < len; i++) {
        exps[i] /= sum;
    }
    return exps;
}

/**
 * Executes forward pass of the Deep Neural Network
 */
function predictDeepLearningIntent(text) {
    if (!window.CHATBOT_METADATA || !window.CHATBOT_WEIGHTS) {
        console.error("Chatbot model weights not loaded!");
        return {
            recognized_text: text,
            intent: "error",
            confidence: 0,
            response: "Model data not loaded. Please ensure model_data.js is included."
        };
    }

    const meta = window.CHATBOT_METADATA;
    const w = window.CHATBOT_WEIGHTS;

    const tokens = cleanAndTokenize(text);
    if (tokens.length === 0) {
        return {
            recognized_text: "",
            intent: "empty",
            confidence: 0,
            response: "No words detected. Please speak clearly into your microphone."
        };
    }

    // 1. Input Bag-of-Words Vector [175]
    let x = bagOfWords(tokens, meta.vocabulary);

    // 2. Layer 1: Linear -> BatchNorm -> ReLU (Dropout disabled during inference)
    x = linear(x, w["fc1.weight"], w["fc1.bias"]);
    x = batchnorm(x, w["bn1.running_mean"], w["bn1.running_var"], w["bn1.weight"], w["bn1.bias"]);
    x = relu(x);

    // 3. Layer 2: Linear -> BatchNorm -> ReLU
    x = linear(x, w["fc2.weight"], w["fc2.bias"]);
    x = batchnorm(x, w["bn2.running_mean"], w["bn2.running_var"], w["bn2.weight"], w["bn2.bias"]);
    x = relu(x);

    // 4. Layer 3: Linear -> ReLU
    x = linear(x, w["fc3.weight"], w["fc3.bias"]);
    x = relu(x);

    // 5. Output Layer: Linear -> Softmax
    x = linear(x, w["out.weight"], w["out.bias"]);
    const probabilities = softmax(x);

    // Find argmax
    let maxIdx = 0;
    let maxProb = -1;
    for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
            maxProb = probabilities[i];
            maxIdx = i;
        }
    }

    const predictedTag = meta.classes[maxIdx];
    const threshold = meta.confidence_threshold || 0.40;

    let finalIntent = predictedTag;
    let responseList = meta.intent_responses[predictedTag];

    if (maxProb < threshold) {
        finalIntent = "low_confidence_fallback";
        responseList = meta.intent_responses["fallback"] || [
            "I recognized your speech, but my deep learning model has low confidence for that phrase. Please try asking about Speech Recognition, Deep Learning, or general queries."
        ];
    }

    // Select random response
    const chosenResponse = responseList[Math.floor(Math.random() * responseList.length)];

    return {
        recognized_text: text,
        intent: finalIntent,
        confidence: Math.round(maxProb * 10000) / 100, // percentage e.g. 96.5%
        response: chosenResponse,
        all_probabilities: meta.classes.map((c, i) => ({ class: c, prob: Math.round(probabilities[i] * 10000) / 100 }))
    };
}
