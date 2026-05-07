import base64
import io
import logging
import os
import tempfile
import time
import uuid
from functools import wraps
from typing import Optional

from flask import Flask, request, jsonify, Response
from model_loader import ModelManager
from model_registry import ModelRegistry
from model_downloader import install_model
from chat_store import ChatStore

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    whisper = None
    HAS_WHISPER = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    pyttsx3 = None
    HAS_TTS = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

metrics = {
    "requests_total": 0,
    "request_duration_seconds": 0.0,
    "chat_completions_total": 0,
    "chat_stream_total": 0,
    "model_operations_total": 0,
    "audio_transcribe_total": 0,
    "audio_speak_total": 0,
}

model_registry = ModelRegistry()
MODEL_ID = os.environ.get("MODEL_ID", "phi-4-multimodal")
if not model_registry.model_exists(MODEL_ID):
    model_registry.register_model(
        MODEL_ID,
        source="huggingface",
        path=MODEL_ID,
        model_type="huggingface",
        description="Default Hugging Face model",
    )
if not model_registry.get_default_model():
    model_registry.set_default_model(MODEL_ID)

model_manager = ModelManager(model_registry)
chat_store = ChatStore()


@app.before_request
def before_request():
    request.start_time = time.time()
    metrics["requests_total"] += 1
    logger.info("Request start: %s %s", request.method, request.path)


@app.after_request
def after_request(response):
    duration = time.time() - getattr(request, "start_time", time.time())
    metrics["request_duration_seconds"] += duration
    logger.info(
        "Request end: %s %s %s %.3fs",
        request.method,
        request.path,
        response.status_code,
        duration,
    )
    return response


@app.errorhandler(Exception)
def handle_exception(error):
    logger.exception("Unhandled exception: %s", error)
    return jsonify({"error": {"message": "Internal server error."}}), 500


def get_request_api_key():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    return request.headers.get("X-API-Key") or request.args.get("api_key")


def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if API_KEY is None:
            return f(*args, **kwargs)
        token = get_request_api_key()
        if token != API_KEY:
            return jsonify({"error": {"message": "Unauthorized"}}), 401
        return f(*args, **kwargs)
    return wrapper


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "model_count": len(model_registry.list_models()),
        "chat_count": len(chat_store.list_chats()),
        "default_model": model_registry.get_default_model(),
    })


@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    lines = []
    for key, value in metrics.items():
        if isinstance(value, int):
            lines.append(f"ai_run_{key} {value}")
        else:
            lines.append(f"ai_run_{key} {value:.6f}")
    return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4")


@app.route("/v1/models", methods=["GET"])
@require_api_key
def list_models():
    return jsonify({
        "object": "list",
        "data": model_registry.list_models(),
        "default_model": model_registry.get_default_model(),
    })


@app.route("/v1/models", methods=["POST"])
@require_api_key
def register_model():
    payload = request.get_json(force=True)
    model_id = payload.get("model_id")
    source = payload.get("source", "huggingface")
    path = payload.get("path")
    model_type = payload.get("model_type", payload.get("type", "huggingface"))
    description = payload.get("description", "")
    install = bool(payload.get("install", False))
    revision = payload.get("revision")
    destination = payload.get("destination")
    token = payload.get("hf_token") or HUGGINGFACE_TOKEN

    if not model_id:
        return jsonify({"error": {"message": "`model_id` is required."}}), 400

    if install:
        if source != "huggingface":
            return jsonify({"error": {"message": "Only Hugging Face installs are supported."}}), 400
        path = install_model(model_id, destination=destination, revision=revision, token=token)

    path = path or model_id
    model_registry.register_model(
        model_id,
        source=source,
        path=path,
        model_type=model_type,
        description=description,
    )

    return jsonify({
        "model_id": model_id,
        "source": source,
        "path": path,
        "model_type": model_type,
        "description": description,
    })


@app.route("/v1/models/<model_id>", methods=["GET"])
@require_api_key
def get_model(model_id):
    model = model_registry.get_model(model_id)
    if not model:
        return jsonify({"error": {"message": "Model not found."}}), 404
    return jsonify(model)


@app.route("/v1/models/<model_id>", methods=["DELETE"])
@require_api_key
def delete_model(model_id):
    if not model_registry.model_exists(model_id):
        return jsonify({"error": {"message": "Model not found."}}), 404
    model_registry.delete_model(model_id)
    return jsonify({"deleted": model_id})


@app.route("/v1/models/default", methods=["POST"])
@require_api_key
def set_default_model():
    payload = request.get_json(force=True)
    model_id = payload.get("model_id")
    if not model_id:
        return jsonify({"error": {"message": "`model_id` is required."}}), 400
    if not model_registry.model_exists(model_id):
        return jsonify({"error": {"message": "Model not registered."}}), 404
    model_registry.set_default_model(model_id)
    return jsonify({"default_model": model_id})


def load_whisper_model():
    if not HAS_WHISPER:
        raise RuntimeError("Audio transcription requires whisper. Install it in requirements.")
    if not hasattr(load_whisper_model, "model"):
        load_whisper_model.model = whisper.load_model("small")
    return load_whisper_model.model


def transcribe_audio_file(file_path: str) -> str:
    model = load_whisper_model()
    result = model.transcribe(file_path)
    return result.get("text", "")


def initialize_tts_engine():
    if not HAS_TTS:
        raise RuntimeError("Text-to-speech requires pyttsx3. Install it in requirements.")
    if not hasattr(initialize_tts_engine, "engine"):
        initialize_tts_engine.engine = pyttsx3.init()
    return initialize_tts_engine.engine


def synthesize_speech(text: str, output_path: str):
    engine = initialize_tts_engine()
    engine.save_to_file(text, output_path)
    engine.runAndWait()


@app.route("/v1/audio/transcribe", methods=["POST"])
@require_api_key
def audio_transcribe():
    metrics["audio_transcribe_total"] += 1
    if "audio" not in request.files:
        return jsonify({"error": {"message": "Missing audio file field 'audio'."}}), 400

    file = request.files["audio"]
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        text = transcribe_audio_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return jsonify({"text": text})


@app.route("/v1/audio/speak", methods=["POST"])
@require_api_key
def audio_speak():
    metrics["audio_speak_total"] += 1
    payload = request.get_json(force=True)
    text = payload.get("text")
    if not text:
        return jsonify({"error": {"message": "`text` is required."}}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    try:
        synthesize_speech(text, output_path)
        with open(output_path, "rb") as fh:
            audio_data = fh.read()
        encoded = base64.b64encode(audio_data).decode("utf-8")
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass

    return jsonify({"audio_base64": encoded})


@app.route("/v1/chats", methods=["GET"])
@require_api_key
def list_chats():
    return jsonify({"object": "list", "data": chat_store.list_chats()})


@app.route("/v1/chat/<chat_id>", methods=["GET"])
@require_api_key
def get_chat(chat_id):
    if not chat_store.chat_exists(chat_id):
        return jsonify({"error": {"message": "Chat not found."}}), 404
    return jsonify({"chat_id": chat_id, "messages": chat_store.get_messages(chat_id)})


@app.route("/v1/chat/<chat_id>", methods=["DELETE"])
@require_api_key
def delete_chat(chat_id):
    if not chat_store.chat_exists(chat_id):
        return jsonify({"error": {"message": "Chat not found."}}), 404
    chat_store.delete_chat(chat_id)
    return jsonify({"deleted": chat_id})


@app.route("/v1/chat/completions", methods=["POST"])
@require_api_key
def chat_completions():
    payload = request.get_json(force=True)
    model_id = payload.get("model", model_registry.get_default_model())
    messages = payload.get("messages", [])
    max_tokens = int(payload.get("max_tokens", 512))
    temperature = float(payload.get("temperature", 0.8))
    top_p = float(payload.get("top_p", 0.95))
    new_chat = bool(payload.get("new_chat", False))
    chat_id = payload.get("chat_id")

    if not model_registry.model_exists(model_id):
        return jsonify({"error": {"message": "Model not registered."}}), 404

    if new_chat or not chat_id:
        chat_id = str(uuid.uuid4())
        chat_store.create_session(chat_id)
    elif not chat_store.chat_exists(chat_id):
        chat_store.create_session(chat_id)

    if not messages or not isinstance(messages, list):
        return jsonify({"error": {"message": "`messages` must be a non-empty list."}}), 400

    chat_store.add_messages(chat_id, messages)
    all_messages = chat_store.get_messages(chat_id)
    prompt = model_manager.format_messages_to_prompt(all_messages)
    output = model_manager.generate(
        model_id=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    assistant_message = {"role": "assistant", "content": output}
    chat_store.add_messages(chat_id, [assistant_message])

    response = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "chat_id": chat_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(output.split()),
            "total_tokens": len((prompt + " " + output).split()),
        },
    }

    return jsonify(response)


@app.route("/v1/chat/stream", methods=["POST"])
@require_api_key
def chat_stream():
    payload = request.get_json(force=True)
    model_id = payload.get("model", model_registry.get_default_model())
    messages = payload.get("messages", [])
    max_tokens = int(payload.get("max_tokens", 512))
    temperature = float(payload.get("temperature", 0.8))
    top_p = float(payload.get("top_p", 0.95))
    new_chat = bool(payload.get("new_chat", False))
    chat_id = payload.get("chat_id")

    if not model_registry.model_exists(model_id):
        return jsonify({"error": {"message": "Model not registered."}}), 404

    if new_chat or not chat_id:
        chat_id = str(uuid.uuid4())
        chat_store.create_session(chat_id)
    elif not chat_store.chat_exists(chat_id):
        chat_store.create_session(chat_id)

    if not messages or not isinstance(messages, list):
        return jsonify({"error": {"message": "`messages` must be a non-empty list."}}), 400

    chat_store.add_messages(chat_id, messages)
    all_messages = chat_store.get_messages(chat_id)
    prompt = model_manager.format_messages_to_prompt(all_messages)

    def event_stream():
        output = ""
        for chunk in model_manager.generate_stream(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            output += chunk
            yield f"data: {chunk}\n\n"

        assistant_message = {"role": "assistant", "content": output}
        chat_store.add_messages(chat_id, [assistant_message])
        yield "event: done\ndata: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/")
def index():
    api_key_required = bool(API_KEY)
    default_model = model_registry.get_default_model()
    api_status = "required" if api_key_required else "not required"
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI-run Chat</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; }
    header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
    section { margin-bottom: 24px; }
    #chat-log { border: 1px solid #ccc; padding: 12px; min-height: 240px; margin-bottom: 12px; white-space: pre-wrap; background: #f9f9f9; }
    .chat-message { margin-bottom: 12px; }
    .role-user { color: #0b6efd; }
    .role-assistant { color: #198754; }
    textarea { width: 100%; height: 80px; }
    button { margin-right: 8px; }
    input, select { padding: 6px; margin-right: 8px; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AI-run Chat</h1>
      <p>Default model: <strong>{{DEFAULT_MODEL}}</strong></p>
      <p>API auth: <strong>{{API_STATUS}}</strong></p>
    </div>
  </header>

  <section>
    <h2>Session</h2>
    <button id="new-chat-btn">Create New Chat</button>
    <select id="chat-select"></select>
    <button id="delete-chat-btn">Delete Chat</button>
  </section>

  <section>
    <h2>Model Catalog</h2>
    <div>
      <select id="model-select"></select>
      <button id="set-default-btn">Set Default</button>
      <button id="delete-model-btn">Delete Model</button>
    </div>
    <div id="model-details"></div>
    <form id="install-form">
      <h3>Install / Register Model</h3>
      <label>Model ID: <input id="install-model-id" required /></label>
      <label>Type: <select id="install-type"><option value="huggingface">huggingface</option><option value="gguf">gguf</option></select></label>
      <label>Destination: <input id="install-destination" placeholder="models/<model_id>" /></label>
      <label>Revision: <input id="install-revision" placeholder="main" /></label>
      <label>Description: <input id="install-description" placeholder="Optional description" /></label>
      <label>Install: <input type="checkbox" id="install-flag" checked /></label>
      <label>HF Token: <input id="install-hf-token" placeholder="Optional token" /></label>
      <button type="submit">Install / Register</button>
    </form>
  </section>

  <section>
    <h2>Chat</h2>
    <div id="chat-log"></div>
    <form id="message-form">
      <textarea id="user-input" placeholder="Type your message here..."></textarea>
      <button type="submit">Send</button>
    </form>
  </section>

  <section>
    <h2>Audio</h2>
    <form id="audio-transcribe-form" enctype="multipart/form-data">
      <label>Upload audio file: <input id="audio-file" type="file" accept="audio/*" /></label>
      <button type="submit">Transcribe</button>
    </form>
    <div id="audio-transcribe-result"></div>

    <form id="audio-speak-form">
      <label>Text to speak: <input id="audio-text" type="text" placeholder="Enter text to synthesize" /></label>
      <button type="submit">Generate Audio</button>
    </form>
    <audio id="audio-player" controls style="display:none;" />
  </section>

  <section>
    <h2>Settings</h2>
    <label>API Token: <input id="api-key-input" type="password" placeholder="Bearer token" /></label>
    <button id="save-api-key-btn">Save Token</button>
  </section>

  <script>
    const chatSelect = document.getElementById('chat-select');
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const newChatBtn = document.getElementById('new-chat-btn');
    const deleteChatBtn = document.getElementById('delete-chat-btn');
    const form = document.getElementById('message-form');
    const modelSelect = document.getElementById('model-select');
    const modelDetails = document.getElementById('model-details');
    const setDefaultBtn = document.getElementById('set-default-btn');
    const deleteModelBtn = document.getElementById('delete-model-btn');
    const installForm = document.getElementById('install-form');
    const apiKeyInput = document.getElementById('api-key-input');
    const saveApiKeyBtn = document.getElementById('save-api-key-btn');
    let activeChatId = null;
    let newChatMode = false;
    let apiToken = '';

    function authHeaders() {
      const headers = { 'Content-Type': 'application/json' };
      if (apiToken) {
        headers['Authorization'] = `Bearer ${apiToken}`;
      }
      return headers;
    }

    async function fetchJson(path, options = {}) {
      options.headers = { ...(options.headers || {}), ...authHeaders() };
      const res = await fetch(path, options);
      return res.json();
    }

    async function loadChats() {
      const data = await fetchJson('/v1/chats');
      chatSelect.innerHTML = '';
      data.data.forEach(chat => {
        const option = document.createElement('option');
        option.value = chat.chat_id;
        option.textContent = `${chat.chat_id} (${new Date(chat.created_at).toLocaleString()})`;
        chatSelect.append(option);
      });
      if (chatSelect.options.length) {
        activeChatId = chatSelect.options[0].value;
        chatSelect.value = activeChatId;
        await loadChat(activeChatId);
      } else {
        activeChatId = null;
        chatLog.textContent = 'No chats yet. Create a new chat or send a message.';
      }
    }

    async function loadModels() {
      const data = await fetchJson('/v1/models');
      modelSelect.innerHTML = '';
      data.data.forEach(model => {
        const option = document.createElement('option');
        option.value = model.model_id;
        option.textContent = `${model.model_id}${model.model_id === data.default_model ? ' (default)' : ''}`;
        modelSelect.append(option);
      });
      if (modelSelect.options.length) {
        modelSelect.value = data.default_model;
        renderModelDetails(data.default_model, data.data);
      } else {
        modelDetails.textContent = 'No registered models.';
      }
    }

    function renderModelDetails(modelId, models) {
      const model = models.find(m => m.model_id === modelId);
      if (!model) {
        modelDetails.textContent = 'Model details not available.';
        return;
      }
      modelDetails.innerHTML = `<strong>${model.model_id}</strong><br/>Type: ${model.model_type}<br/>Source: ${model.source}<br/>Path: ${model.path}<br/>Description: ${model.description || 'n/a'}`;
    }

    async function loadChat(chatId) {
      const data = await fetchJson(`/v1/chat/${chatId}`);
      if (data.error) {
        chatLog.textContent = 'Unable to load chat.';
        return;
      }
      activeChatId = data.chat_id;
      newChatMode = false;
      chatLog.innerHTML = data.messages.map(msg => `<div class="chat-message"><strong class="role-${msg.role}">${msg.role}:</strong> ${msg.content}</div>`).join('');
    }

    newChatBtn.addEventListener('click', () => {
      activeChatId = null;
      newChatMode = true;
      chatLog.textContent = 'New chat ready. Type a message and send to begin.';
    });

    chatSelect.addEventListener('change', async () => {
      activeChatId = chatSelect.value;
      newChatMode = false;
      await loadChat(activeChatId);
    });

    deleteChatBtn.addEventListener('click', async () => {
      if (!activeChatId) return;
      await fetchJson(`/v1/chat/${activeChatId}`, { method: 'DELETE' });
      await loadChats();
    });

    setDefaultBtn.addEventListener('click', async () => {
      await fetchJson('/v1/models/default', {
        method: 'POST',
        body: JSON.stringify({ model_id: modelSelect.value }),
      });
      await loadModels();
    });

    deleteModelBtn.addEventListener('click', async () => {
      const modelId = modelSelect.value;
      if (!modelId) return;
      await fetchJson(`/v1/models/${modelId}`, { method: 'DELETE' });
      await loadModels();
    });

    installForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const modelId = document.getElementById('install-model-id').value.trim();
      const modelType = document.getElementById('install-type').value;
      const destination = document.getElementById('install-destination').value.trim();
      const revision = document.getElementById('install-revision').value.trim();
      const description = document.getElementById('install-description').value.trim();
      const install = document.getElementById('install-flag').checked;
      const hfToken = document.getElementById('install-hf-token').value.trim();

      const body = {
        model_id: modelId,
        source: 'huggingface',
        model_type: modelType,
        description,
        install,
      };
      if (destination) body.destination = destination;
      if (revision) body.revision = revision;
      if (hfToken) body.hf_token = hfToken;

      await fetchJson('/v1/models', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      await loadModels();
    });

    document.getElementById('audio-transcribe-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const fileInput = document.getElementById('audio-file');
      if (!fileInput.files.length) return;
      const formData = new FormData();
      formData.append('audio', fileInput.files[0]);
      const res = await fetch('/v1/audio/transcribe', {
        method: 'POST',
        headers: apiToken ? { 'Authorization': `Bearer ${apiToken}` } : {},
        body: formData,
      });
      const data = await res.json();
      document.getElementById('audio-transcribe-result').textContent = data.text || data.error?.message || 'Transcription failed.';
    });

    document.getElementById('audio-speak-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const text = document.getElementById('audio-text').value.trim();
      if (!text) return;
      const data = await fetchJson('/v1/audio/speak', {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      if (data.audio_base64) {
        const audioPlayer = document.getElementById('audio-player');
        audioPlayer.src = `data:audio/wav;base64,${data.audio_base64}`;
        audioPlayer.style.display = 'block';
        audioPlayer.load();
      }
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const content = userInput.value.trim();
      if (!content) return;
      const body = {
        model: modelSelect.value,
        messages: [{ role: 'user', content }],
      };
      if (activeChatId && !newChatMode) {
        body.chat_id = activeChatId;
      } else {
        body.new_chat = true;
      }
      const data = await fetchJson('/v1/chat/completions', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      if (data.chat_id) {
        activeChatId = data.chat_id;
        newChatMode = false;
        await loadChats();
        chatLog.innerHTML += `<div class="chat-message"><strong class="role-user">user:</strong> ${content}</div><div class="chat-message"><strong class="role-assistant">assistant:</strong> ${data.choices[0].message.content}</div>`;
      }
      userInput.value = '';
    });

    saveApiKeyBtn.addEventListener('click', () => {
      apiToken = apiKeyInput.value.trim();
      alert('API token saved for this browser session.');
    });

    loadModels();
    loadChats();
  </script>
</body>
</html>
"""
    return html.replace("{{DEFAULT_MODEL}}", default_model).replace("{{API_STATUS}}", api_status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
