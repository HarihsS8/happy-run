# AI-run

A local LLM inference engine, model registry, streaming API, and secure chat server.

## Features

- Model registry for Hugging Face models and local `gguf`/custom model paths
- Model install/download tooling via Hugging Face
- Persistent SQLite storage for chat sessions and model metadata
- API key protection via `API_KEY`
- OpenAI-compatible chat completions and SSE streaming
- Browser UI for chat, model catalog, installation, and default model management
- CLI tooling for model and chat session administration

## Quick start

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Configure security and optional Hugging Face access:

```bash
export API_KEY="your-secret-token"
export HUGGINGFACE_TOKEN="your-hf-token"
```

3. Start the server:

```bash
python app.py
```

4. Open the browser UI:

```bash
http://localhost:8080/
```

5. Install a model from Hugging Face:

```bash
python cli.py model install phi-4-multimodal --description "Default local model"
```

6. Set the default model:

```bash
python cli.py model default phi-4-multimodal
```

## CLI

- `python cli.py model list`
- `python cli.py model install <model_id>`
- `python cli.py model add <model_id> --path <local-path>`
- `python cli.py model default <model_id>`
- `python cli.py model delete <model_id>`
- `python cli.py chat list`
- `python cli.py chat delete <chat_id>`
- `python cli.py run --new-chat`

## Endpoints

- `GET /v1/models` — list registered models and default model
- `POST /v1/models` — register or install a model
- `GET /v1/models/<model_id>` — retrieve model metadata
- `DELETE /v1/models/<model_id>` — remove a registered model
- `POST /v1/models/default` — set the default model
- `GET /v1/chats` — list saved chat sessions
- `GET /v1/chat/<chat_id>` — retrieve chat history
- `DELETE /v1/chat/<chat_id>` — delete a chat session
- `POST /v1/chat/completions` — generate a chat completion (supports tools for agent compatibility)
- `POST /v1/chat/stream` — stream completion chunks via SSE (supports tools)
- `POST /v1/embeddings` — generate text embeddings for a model

## Notes

- The default model is stored in the registry and initialized as `phi-4-multimodal` on first run.
- Model installation downloads a local copy to `models/<model_id>` by default.
- Chat sessions are persisted to `chat_history.db`.
- Model metadata is persisted to `model_registry.db`.
- If `API_KEY` is set, all `/v1/*` endpoints require `Authorization: Bearer <token>`.
- If `HUGGINGFACE_TOKEN` is set, installs may access private Hugging Face repos.
- Chat completions support `tools` parameter for AI agent compatibility (JSON tool calling).
- Running larger models may require GPU memory and additional setup.
