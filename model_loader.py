import threading
from typing import Any, Dict, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import TextIteratorStreamer
    HAS_STREAMER = True
except ImportError:
    TextIteratorStreamer = None
    HAS_STREAMER = False

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    Llama = None
    HAS_LLAMA_CPP = False


class ModelManager:
    def __init__(self, registry):
        self.registry = registry
        self.loaded_models: dict[str, dict[str, Any]] = {}

    def load_model(self, model_id: str) -> dict[str, Any]:
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        model_info = self.registry.get_model(model_id) or {
            "model_id": model_id,
            "path": model_id,
            "model_type": "huggingface",
            "source": "huggingface",
        }
        path = model_info["path"]
        model_type = model_info["model_type"].lower()

        if model_type == "gguf" or str(path).lower().endswith(".gguf"):
            return self._load_gguf(model_id, path)

        return self._load_transformers(model_id, path)

    def _load_transformers(self, model_id: str, path: str) -> dict[str, Any]:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

        self.loaded_models[model_id] = {
            "type": "transformers",
            "model": model,
            "tokenizer": tokenizer,
        }
        return self.loaded_models[model_id]

    def _load_gguf(self, model_id: str, path: str) -> dict[str, Any]:
        if not HAS_LLAMA_CPP:
            raise RuntimeError("GGUF models require llama-cpp-python. Install it in requirements.")

        llm = Llama(model_path=path)
        self.loaded_models[model_id] = {
            "type": "gguf",
            "llm": llm,
        }
        return self.loaded_models[model_id]

    def format_messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        prompt_lines = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_lines.append(f"[SYSTEM]: {content}")
            elif role == "assistant":
                prompt_lines.append(f"[ASSISTANT]: {content}")
            else:
                prompt_lines.append(f"[USER]: {content}")
        prompt_lines.append("[ASSISTANT]:")
        return "\n".join(prompt_lines)

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        model_data = self.load_model(model_id)
        if model_data["type"] == "gguf":
            return self._generate_gguf(model_data, prompt, max_tokens, temperature, top_p)

        return self._generate_transformers(model_data, prompt, max_tokens, temperature, top_p)

    def generate_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Iterator[str]:
        model_data = self.load_model(model_id)
        if model_data["type"] == "gguf":
            text = self._generate_gguf(model_data, prompt, max_tokens, temperature, top_p)
            yield from self._chunk_text(text)
            return

        if HAS_STREAMER:
            yield from self._stream_transformers(model_data, prompt, max_tokens, temperature, top_p)
            return

        text = self._generate_transformers(model_data, prompt, max_tokens, temperature, top_p)
        yield from self._chunk_text(text)

    def _generate_transformers(
        self,
        model_data: dict[str, Any],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        result = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(result[0], skip_special_tokens=True).strip()

    def _generate_gguf(
        self,
        model_data: dict[str, Any],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        llm = model_data["llm"]
        completion = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return completion["choices"][0]["text"].strip()

    def _stream_transformers(
        self,
        model_data: dict[str, Any],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        if not HAS_STREAMER:
            raise RuntimeError("Streaming is not available because TextIteratorStreamer is missing.")

        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        full_text = ""
        for chunk in streamer:
            full_text += chunk
            yield chunk

        thread.join()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 80) -> Iterator[str]:
        for index in range(0, len(text), chunk_size):
            yield text[index : index + chunk_size]
