import random
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict
from zipfile import ZipFile

import requests

from chronoseek.validator.base_task_gen import BaseTaskGenerator


class ActivityNetTaskGenerator(BaseTaskGenerator):
    """
    Generates tasks based on the ActivityNet Captions dataset (MVP Scope).
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        split: str = "validation",
        dataset_repo_id: str = "friedrichor/ActivityNet_Captions",
        cache_dir: str | None = None,
        dataset_filename: str | None = None,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.dataset_repo_id = dataset_repo_id
        self.cache_dir = cache_dir or os.getenv("HF_HOME")
        self.dataset_filename = dataset_filename or os.getenv("HF_ACTIVITYNET_FILENAME")
        self.dataset = self._load_dataset()

    def _default_dataset_path(self) -> str:
        return str(Path(__file__).resolve().parent / "data" / "activitynet_bootstrap.json")

    def _load_huggingface_dataset(self) -> List[Dict]:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN is required for validator task generation from Hugging Face."
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "The `huggingface_hub` package is required to download ActivityNet from Hugging Face."
            ) from exc

        snapshot_dir = snapshot_download(
            repo_id=self.dataset_repo_id,
            repo_type="dataset",
            token=hf_token,
            cache_dir=self.cache_dir,
        )

        dataset_file = self._resolve_snapshot_dataset_file(snapshot_dir)
        return self._load_local_dataset(dataset_file)

    def _resolve_snapshot_dataset_file(self, snapshot_dir: str) -> str:
        root = Path(snapshot_dir)

        if self.dataset_filename:
            candidate = root / self.dataset_filename
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured ActivityNet file '{self.dataset_filename}' was not found in Hugging Face snapshot."
                )
            return str(candidate)

        json_candidates = sorted(root.rglob("*.json"))
        for candidate in json_candidates:
            try:
                with candidate.open("r") as handle:
                    data = json.load(handle)
            except Exception:
                continue

            if isinstance(data, dict) and ("tasks" in data or "database" in data):
                return str(candidate)

        dataset_script = root / "ActivityNet_Captions.py"
        if dataset_script.exists():
            return self._download_original_activitynet_split(root)

        raise FileNotFoundError(
            f"No supported ActivityNet JSON file was found in the Hugging Face snapshot for {self.dataset_repo_id}."
        )

    def _download_original_activitynet_split(self, snapshot_root: Path) -> str:
        source_url = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"
        cache_root = Path(self.cache_dir) if self.cache_dir else snapshot_root.parent.parent
        activitynet_cache = cache_root / "chronoseek-activitynet"
        activitynet_cache.mkdir(parents=True, exist_ok=True)

        archive_path = activitynet_cache / "captions.zip"
        extract_dir = activitynet_cache / "captions"

        if not archive_path.exists():
            response = requests.get(source_url, stream=True, timeout=120)
            response.raise_for_status()
            with archive_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)

        split_filename = {
            "train": "train.json",
            "validation": "val_1.json",
            "test": "val_2.json",
        }.get(self.split)

        if split_filename is None:
            raise ValueError(
                f"Unsupported ActivityNet split '{self.split}'. Expected train, validation, or test."
            )

        target_path = extract_dir / split_filename
        if not target_path.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_dir)

        if not target_path.exists():
            raise FileNotFoundError(
                f"Expected ActivityNet split file '{split_filename}' was not found after extracting captions.zip."
            )

        return str(target_path)

    def _normalize_manifest_tasks(self, data: dict) -> List[Dict]:
        tasks = []
        for row in data["tasks"]:
            if row.get("split") != self.split:
                continue

            tasks.append(
                {
                    "task_id": row.get("task_id"),
                    "split": row.get("split"),
                    "difficulty": row.get("difficulty"),
                    "video_url": row["video_url"],
                    "captions": [row["query"]],
                    "ground_truth_starts": [float(row["ground_truth"]["start"])],
                    "ground_truth_ends": [float(row["ground_truth"]["end"])],
                }
            )
        return tasks

    def _normalize_activitynet_database(self, data: dict) -> List[Dict]:
        tasks = []
        for vid_id, content in data.get("database", {}).items():
            url = content.get("url", "")
            sentences = content.get("sentences", [])
            timestamps = content.get("timestamps", [])
            if not url or not sentences or len(sentences) != len(timestamps):
                continue

            tasks.append(
                {
                    "task_id": vid_id,
                    "split": self.split,
                    "difficulty": "unknown",
                    "video_url": url,
                    "captions": list(sentences),
                    "ground_truth_starts": [float(pair[0]) for pair in timestamps],
                    "ground_truth_ends": [float(pair[1]) for pair in timestamps],
                }
            )
        return tasks

    def _load_local_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load a local ActivityNet manifest for offline use or tests.
        """
        with open(dataset_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "tasks" in data:
                tasks = self._normalize_manifest_tasks(data)
                if tasks:
                    return tasks

            tasks = self._normalize_activitynet_database(data)
            if tasks:
                return tasks

        raise ValueError(f"No usable ActivityNet tasks were found in {dataset_path}")

    def _load_dataset(self) -> List[Dict]:
        if self.dataset_path:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(
                    f"ActivityNet task dataset not found at {self.dataset_path}"
                )
            return self._load_local_dataset(self.dataset_path)

        return self._load_huggingface_dataset()

    def generate_task(self) -> Tuple[str, str, Tuple[float, float]]:
        """
        Returns: (video_url, query, ground_truth_interval)
        """
        video = random.choice(self.dataset)
        caption_index = random.randrange(len(video["captions"]))

        return (
            video["video_url"],
            video["captions"][caption_index],
            (
                video["ground_truth_starts"][caption_index],
                video["ground_truth_ends"][caption_index],
            ),
        )
