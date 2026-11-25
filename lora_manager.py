import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


logger = logging.getLogger("klein-server.lora")


@dataclass(frozen=True)
class LoraCatalogItem:
    index: int
    repo_id: str
    downloads: int
    likes: int
    updated_at: str | None
    pipeline_tag: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoraManager:
    """Catalogo e cache locale dei LoRA compatibili con un modello base."""

    def __init__(
        self,
        base_model_id: str,
        cache_dir: str | None = None,
        catalog_ttl: int = 300,
    ) -> None:
        self.base_model_id = base_model_id
        self.cache_dir = cache_dir or os.environ.get("KLEIN_LORA_CACHE_DIR")
        self.catalog_ttl = catalog_ttl
        self._api = HfApi()
        self._catalog: list[LoraCatalogItem] = []
        self._catalog_updated_at = 0.0
        self._catalog_lock = threading.RLock()

    @property
    def hub_filter(self) -> str:
        return f"base_model:adapter:{self.base_model_id}"

    def list_catalog(self, refresh: bool = False) -> list[dict[str, Any]]:
        with self._catalog_lock:
            expired = time.monotonic() - self._catalog_updated_at >= self.catalog_ttl
            if refresh or not self._catalog or expired:
                models = self._api.list_models(
                    filter=self.hub_filter,
                    sort="downloads",
                    full=True,
                )
                items = []
                for index, model in enumerate(models):
                    updated_at = getattr(model, "last_modified", None)
                    if hasattr(updated_at, "isoformat"):
                        updated_at = updated_at.isoformat()
                    items.append(
                        LoraCatalogItem(
                            index=index,
                            repo_id=model.id,
                            downloads=getattr(model, "downloads", 0) or 0,
                            likes=getattr(model, "likes", 0) or 0,
                            updated_at=updated_at,
                            pipeline_tag=getattr(model, "pipeline_tag", None),
                        )
                    )
                self._catalog = items
                self._catalog_updated_at = time.monotonic()
                logger.info("Catalogo LoRA aggiornato: %d adapter", len(items))
            return [item.to_dict() for item in self._catalog]

    def get(self, index: int) -> LoraCatalogItem:
        with self._catalog_lock:
            catalog = self.list_catalog()
            if index < 0 or index >= len(catalog):
                raise IndexError(
                    f"Indice LoRA {index} non valido; intervallo disponibile: "
                    f"0..{max(len(catalog) - 1, 0)}"
                )
            return self._catalog[index]

    def download(
        self,
        index: int,
        *,
        revision: str | None = None,
    ) -> tuple[LoraCatalogItem, Path]:
        item = self.get(index)
        local_path = snapshot_download(
            repo_id=item.repo_id,
            revision=revision,
            cache_dir=self.cache_dir,
            allow_patterns=["*.safetensors", "*.json", "**/*.safetensors", "**/*.json"],
        )
        return item, Path(local_path)

    def download_repo(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
    ) -> Path:
        """Scarica un adapter indicato direttamente, senza passare dal catalogo."""
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=self.cache_dir,
            allow_patterns=["*.safetensors", "*.json", "**/*.safetensors", "**/*.json"],
        )
        return Path(local_path)

    def declared_base_models(self, repo_id: str) -> list[str]:
        """Restituisce i modelli base dichiarati nella model card dell'adapter."""
        info = self._api.model_info(repo_id)
        card_data = getattr(info, "card_data", None)
        base_model = getattr(card_data, "base_model", None)
        if base_model is None and isinstance(card_data, dict):
            base_model = card_data.get("base_model")
        if isinstance(base_model, str):
            return [base_model]
        if isinstance(base_model, list):
            return [item for item in base_model if isinstance(item, str)]
        return []
