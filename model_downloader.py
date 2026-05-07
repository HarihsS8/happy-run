import os
from huggingface_hub import snapshot_download


def install_model(
    model_id: str,
    destination: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    destination = destination or os.path.join("models", model_id)
    os.makedirs(destination, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=model_id,
        cache_dir=destination,
        revision=revision,
        use_auth_token=token,
    )
    return os.path.abspath(downloaded_path)
