import random
import json
import os
from collections import defaultdict
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
        return str(Path(__file__).resolve().parent / "data" / "smoke_test_tasks.json")

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

            if self._is_supported_dataset_payload(data):
                return str(candidate)

        parquet_candidates = sorted(root.rglob("*.parquet"))
        if parquet_candidates:
            return str(parquet_candidates[0])

        dataset_script = root / "ActivityNet_Captions.py"
        if dataset_script.exists():
            return self._download_original_activitynet_split(root)

        raise FileNotFoundError(
            f"No supported ActivityNet JSON file was found in the Hugging Face snapshot for {self.dataset_repo_id}."
        )

    def _is_supported_dataset_payload(self, data) -> bool:
        if isinstance(data, list):
            return True

        if not isinstance(data, dict):
            return False

        if any(key in data for key in ("tasks", "database", "rows", "data")):
            return True

        return any(isinstance(value, dict) for value in data.values())

    def _download_original_activitynet_split(self, snapshot_root: Path) -> str:
        source_url = (
            "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"
        )
        cache_root = (
            Path(self.cache_dir) if self.cache_dir else snapshot_root.parent.parent
        )
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

            intervals = self._normalize_interval_list(
                row.get("ground_truths")
                if "ground_truths" in row
                else row.get("ground_truth")
            )
            if not intervals:
                continue

            tasks.append(
                {
                    "task_id": row.get("task_id"),
                    "split": row.get("split"),
                    "difficulty": row.get("difficulty"),
                    "video_url": row["video_url"],
                    "caption_intervals": {
                        row["query"]: intervals,
                    },
                }
            )
        return tasks

    def _normalize_interval_list(self, raw_intervals) -> List[Tuple[float, float]]:
        if raw_intervals is None:
            return []

        if isinstance(raw_intervals, dict):
            if "start" in raw_intervals and "end" in raw_intervals:
                return [(float(raw_intervals["start"]), float(raw_intervals["end"]))]
            return []

        if isinstance(raw_intervals, (list, tuple)):
            if len(raw_intervals) == 2 and all(
                isinstance(value, (int, float)) for value in raw_intervals
            ):
                return [(float(raw_intervals[0]), float(raw_intervals[1]))]

            normalized = []
            for item in raw_intervals:
                normalized.extend(self._normalize_interval_list(item))
            return normalized

        return []

    def _normalize_activitynet_database(self, data: dict) -> List[Dict]:
        tasks = []
        database = (
            data.get("database") if isinstance(data.get("database"), dict) else data
        )
        for vid_id, content in database.items():
            if not isinstance(content, dict):
                continue

            url = (
                content.get("url")
                or f"https://www.youtube.com/watch?v={str(vid_id)[2:]}"
            )
            sentences = content.get("sentences", [])
            timestamps = content.get("timestamps", [])
            if not url or not sentences or len(sentences) != len(timestamps):
                continue

            caption_intervals: Dict[str, List[Tuple[float, float]]] = {}
            for sentence, pair in zip(sentences, timestamps):
                if len(pair) != 2:
                    continue
                caption_intervals.setdefault(sentence, []).append(
                    (float(pair[0]), float(pair[1]))
                )

            if not caption_intervals:
                continue

            tasks.append(
                {
                    "task_id": vid_id,
                    "split": self.split,
                    "difficulty": "unknown",
                    "video_url": url,
                    "caption_intervals": caption_intervals,
                }
            )
        return tasks

    def _normalize_activitynet_rows(self, rows: List[dict]) -> List[Dict]:
        grouped: Dict[str, Dict] = {}

        for row in rows:
            if not isinstance(row, dict):
                continue

            split = row.get("split", self.split)
            if split != self.split:
                continue

            video_id = row.get("video_id") or row.get("id") or row.get("task_id")
            video_url = row.get("video_url") or row.get("url")
            if not video_url and video_id:
                video_url = f"https://www.youtube.com/watch?v={str(video_id)[2:]}"

            caption = row.get("caption") or row.get("query") or row.get("sentence")
            intervals = self._normalize_interval_list(
                row.get("ground_truths")
                if "ground_truths" in row
                else (
                    row.get("ground_truth")
                    if "ground_truth" in row
                    else (
                        [row.get("start_time"), row.get("end_time")]
                        if row.get("start_time") is not None
                        and row.get("end_time") is not None
                        else row.get("timestamps")
                    )
                )
            )

            if not video_url or not caption or not intervals:
                continue

            task_key = str(video_id or video_url)
            if task_key not in grouped:
                grouped[task_key] = {
                    "task_id": task_key,
                    "split": split,
                    "difficulty": row.get("difficulty", "unknown"),
                    "video_url": video_url,
                    "caption_intervals": defaultdict(list),
                }

            grouped[task_key]["caption_intervals"][caption].extend(intervals)

        tasks = []
        for task in grouped.values():
            caption_intervals = {
                caption: self._dedupe_intervals(intervals)
                for caption, intervals in task["caption_intervals"].items()
                if intervals
            }
            if not caption_intervals:
                continue

            tasks.append(
                {
                    "task_id": task["task_id"],
                    "split": task["split"],
                    "difficulty": task["difficulty"],
                    "video_url": task["video_url"],
                    "caption_intervals": caption_intervals,
                }
            )
        return tasks

    def _dedupe_intervals(
        self, intervals: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        seen = set()
        deduped: List[Tuple[float, float]] = []
        for start, end in intervals:
            key = (round(float(start), 4), round(float(end), 4))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((float(start), float(end)))
        return deduped

    def _load_local_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load a local ActivityNet manifest for offline use or tests.
        """
        if dataset_path.endswith(".parquet"):
            return self._load_parquet_dataset(dataset_path)

        with open(dataset_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                tasks = self._normalize_activitynet_rows(data)
                if tasks:
                    return tasks

            if isinstance(data, dict) and "tasks" in data:
                tasks = self._normalize_manifest_tasks(data)
                if tasks:
                    return tasks

            if isinstance(data, dict):
                for key in ("rows", "data"):
                    if isinstance(data.get(key), list):
                        tasks = self._normalize_activitynet_rows(data[key])
                        if tasks:
                            return tasks

            tasks = self._normalize_activitynet_database(data)
            if tasks:
                return tasks

        raise ValueError(f"No usable ActivityNet tasks were found in {dataset_path}")

    def _load_parquet_dataset(self, dataset_path: str) -> List[Dict]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to read ActivityNet parquet files from Hugging Face snapshots."
            ) from exc

        table = pq.read_table(dataset_path)
        rows = table.to_pylist()
        tasks = self._normalize_activitynet_rows(rows)
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

    def generate_task(self) -> Tuple[str, str, List[Tuple[float, float]]]:
        """
        Returns: (video_url, query, ground_truth_intervals)
        """
        video = random.choice(self.dataset)
        captions = list(video["caption_intervals"].keys())
        caption = random.choice(captions)

        return (
            video["video_url"],
            caption,
            list(video["caption_intervals"][caption]),
        )
