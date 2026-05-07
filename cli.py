import argparse
import sys
import uuid
from model_registry import ModelRegistry
from model_loader import ModelManager
from model_downloader import install_model
from chat_store import ChatStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-run command line manager")
    subparsers = parser.add_subparsers(dest="command")

    model_parser = subparsers.add_parser("model", help="Manage available models")
    model_subparsers = model_parser.add_subparsers(dest="subcommand")

    model_list = model_subparsers.add_parser("list", help="List registered models")
    model_list.set_defaults(func=handle_model_list)

    model_add = model_subparsers.add_parser("add", help="Register an existing model path or repo")
    model_add.add_argument("model_id", help="Model identifier")
    model_add.add_argument("--path", help="Local path or Hugging Face repo path")
    model_add.add_argument("--type", default="huggingface", choices=["huggingface", "gguf"], help="Model type")
    model_add.add_argument("--source", default="huggingface", help="Model source label")
    model_add.add_argument("--description", default="", help="Optional model description")
    model_add.set_defaults(func=handle_model_add)

    model_install = model_subparsers.add_parser("install", help="Download and register a Hugging Face model")
    model_install.add_argument("model_id", help="Hugging Face repo ID to install")
    model_install.add_argument("--destination", help="Local destination directory")
    model_install.add_argument("--revision", help="Repository revision or branch")
    model_install.add_argument("--type", default="huggingface", choices=["huggingface", "gguf"], help="Model type")
    model_install.add_argument("--description", default="", help="Optional model description")
    model_install.add_argument("--token", help="Hugging Face token for private repos")
    model_install.set_defaults(func=handle_model_install)

    model_default = model_subparsers.add_parser("default", help="Set the default model")
    model_default.add_argument("model_id", help="Model identifier to set as default")
    model_default.set_defaults(func=handle_model_default)

    model_delete = model_subparsers.add_parser("delete", help="Remove a model registration")
    model_delete.add_argument("model_id", help="Model identifier to delete")
    model_delete.set_defaults(func=handle_model_delete)

    chat_parser = subparsers.add_parser("chat", help="Manage saved chat sessions")
    chat_subparsers = chat_parser.add_subparsers(dest="subcommand")

    chat_list = chat_subparsers.add_parser("list", help="List chat sessions")
    chat_list.set_defaults(func=handle_chat_list)

    chat_delete = chat_subparsers.add_parser("delete", help="Delete a chat session")
    chat_delete.add_argument("chat_id", help="Chat session identifier")
    chat_delete.set_defaults(func=handle_chat_delete)

    run_parser = subparsers.add_parser("run", help="Run an interactive chat session")
    run_parser.add_argument("--model", help="Model identifier to use")
    run_parser.add_argument("--new-chat", action="store_true", help="Start a new session")
    run_parser.add_argument("--chat-id", help="Existing chat session ID to continue")
    run_parser.set_defaults(func=handle_run)

    return parser


def handle_model_list(args: argparse.Namespace):
    registry = ModelRegistry()
    models = registry.list_models()
    default_model = registry.get_default_model()
    if not models:
        print("No models registered.")
        return
    for model in models:
        default_marker = " (default)" if model["model_id"] == default_model else ""
        print(f"{model['model_id']}{default_marker}")
        print(f"  type: {model['model_type']}")
        print(f"  source: {model['source']}")
        print(f"  path: {model['path']}")
        if model.get("description"):
            print(f"  description: {model['description']}")
        print()


def handle_model_add(args: argparse.Namespace):
    registry = ModelRegistry()
    path = args.path or args.model_id
    registry.register_model(
        args.model_id,
        source=args.source,
        path=path,
        model_type=args.type,
        description=args.description,
    )
    print(f"Registered model '{args.model_id}' with path '{path}'.")


def handle_model_install(args: argparse.Namespace):
    registry = ModelRegistry()
    local_path = install_model(
        args.model_id,
        destination=args.destination,
        revision=args.revision,
        token=args.token,
    )
    registry.register_model(
        args.model_id,
        source="huggingface",
        path=local_path,
        model_type=args.type,
        description=args.description,
    )
    print(f"Installed and registered model '{args.model_id}' at '{local_path}'.")


def handle_model_default(args: argparse.Namespace):
    registry = ModelRegistry()
    if not registry.model_exists(args.model_id):
        print(f"Model '{args.model_id}' is not registered.")
        sys.exit(1)
    registry.set_default_model(args.model_id)
    print(f"Default model set to '{args.model_id}'.")


def handle_model_delete(args: argparse.Namespace):
    registry = ModelRegistry()
    if not registry.model_exists(args.model_id):
        print(f"Model '{args.model_id}' does not exist.")
        sys.exit(1)
    registry.delete_model(args.model_id)
    print(f"Deleted model '{args.model_id}'.")


def handle_chat_list(args: argparse.Namespace):
    store = ChatStore()
    chats = store.list_chats()
    if not chats:
        print("No saved chats.")
        return
    for chat in chats:
        print(f"{chat['chat_id']} - created {chat['created_at']}")


def handle_chat_delete(args: argparse.Namespace):
    store = ChatStore()
    if not store.chat_exists(args.chat_id):
        print(f"Chat '{args.chat_id}' not found.")
        sys.exit(1)
    store.delete_chat(args.chat_id)
    print(f"Deleted chat '{args.chat_id}'.")


def handle_run(args: argparse.Namespace):
    registry = ModelRegistry()
    model_id = args.model or registry.get_default_model()
    if not model_id:
        print("No model selected and no default model is configured.")
        sys.exit(1)
    if not registry.model_exists(model_id):
        print(f"Model '{model_id}' is not registered.")
        sys.exit(1)

    manager = ModelManager(registry)
    store = ChatStore()
    chat_id = args.chat_id
    if args.new_chat or not chat_id:
        chat_id = str(uuid.uuid4())
        store.create_session(chat_id)
        print(f"Starting new chat session {chat_id}")
    else:
        if not store.chat_exists(chat_id):
            store.create_session(chat_id)
        print(f"Continuing chat session {chat_id}")

    print(f"Using model: {model_id}")
    print("Type a message and press Enter. Type /exit to quit.")

    while True:
        content = input("you> ").strip()
        if content.lower() in {"/exit", "/quit"}:
            break
        if not content:
            continue

        user_message = [{"role": "user", "content": content}]
        store.add_messages(chat_id, user_message)
        all_messages = store.get_messages(chat_id)
        prompt = manager.format_messages_to_prompt(all_messages)
        output = manager.generate(model_id=model_id, prompt=prompt)
        assistant_message = [{"role": "assistant", "content": output}]
        store.add_messages(chat_id, assistant_message)
        print(f"assistant> {output}\n")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
