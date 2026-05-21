import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bittensor as bt


@dataclass(frozen=True)
class QueryVariant:
    variant_id: str
    query: str
    source: str


class QueryVariantSelector:
    """
    Selects private query variants for ActivityNet captions.

    Supported manifest shapes:
    - {"raw caption": ["variant 1", "variant 2"]}
    - {"caption-id": ["variant 1", "variant 2"]}
    - [{"caption": "...", "variants": [...]}, ...]
    - [{"source_caption_id": "...", "variants": [...]}, ...]
    """

    def __init__(self, path: str | None = None):
        self.path = str(Path(path).expanduser()) if path else None
        self._variants: dict[str, list[str]] = {}
        if self.path:
            self._variants = self._load_manifest(self.path)
            bt.logging.info(
                f"Loaded {sum(len(v) for v in self._variants.values())} private query variants from {self.path}."
            )

    @staticmethod
    def _coerce_variants(raw_variants: Any) -> list[str]:
        if isinstance(raw_variants, str):
            raw_variants = [raw_variants]
        if not isinstance(raw_variants, list):
            return []

        variants = []
        for variant in raw_variants:
            if isinstance(variant, str) and variant.strip():
                variants.append(variant.strip())
        return variants

    def _load_manifest(self, path: str) -> dict[str, list[str]]:
        manifest_path = Path(path).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Query variant manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        variants: dict[str, list[str]] = {}
        if isinstance(payload, dict):
            for key, raw_variants in payload.items():
                normalized = self._coerce_variants(raw_variants)
                if normalized:
                    variants[str(key)] = normalized
            return variants

        if isinstance(payload, list):
            for index, row in enumerate(payload):
                if not isinstance(row, dict):
                    continue
                key = (
                    row.get("source_caption_id")
                    or row.get("caption_id")
                    or row.get("caption")
                    or row.get("source_caption")
                )
                normalized = self._coerce_variants(row.get("variants"))
                if key and normalized:
                    variants[str(key)] = normalized
                elif normalized:
                    variants[f"row:{index}"] = normalized
            return variants

        raise ValueError(
            "Query variant manifest must be a dictionary or a list of rows."
        )

    @staticmethod
    def _fallback_variants(caption: str) -> list[str]:
        caption = caption.strip()
        return [
            f"Find the moment where this happens: {caption}",
            f"Locate the scene that best matches this description: {caption}",
            f"When does the video show the following event? {caption}",
        ]

    def select(
        self,
        *,
        caption: str,
        source_caption_id: str,
        rng: random.Random,
    ) -> QueryVariant:
        candidates = (
            self._variants.get(source_caption_id)
            or self._variants.get(caption)
            or []
        )
        source = "private"
        if not candidates:
            candidates = self._fallback_variants(caption)
            source = "fallback"
            bt.logging.warning(
                "No private query variant was found for sampled ActivityNet caption; using deterministic fallback wording."
            )

        index = rng.randrange(len(candidates))
        query = candidates[index]
        variant_id = f"{source}:{source_caption_id}:{index}"
        return QueryVariant(variant_id=variant_id, query=query, source=source)
